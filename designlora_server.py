"""
DesignLoRA server.

FastAPI service implementing the DesignLoRA designer-in-the-loop loop:
LoRA-based generation, dual-prompt (question + preference) VLM evaluation,
VLM-guided gradient distillation, and Expected-Improvement candidate sampling.

Built on top of Dual-Process Image Generation (Luo et al., ICCV 2025);
see dual_process/UPSTREAM_CHANGES.md for what DesignLoRA modifies.
"""

import base64
import io
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# DesignLoRA modules (built on the dual_process base; see dual_process/UPSTREAM_CHANGES.md).
from dual_process import dig_helpers, dig_pipeline, dig_operators, llm_surrogate
from vlm_analysis_utils import describe_image, _update_preference_profile, setup_vlm_context
from logging_utils import get_logger

# ============= Configuration =============
# Output locations are overridable via environment variables so the repo ships no machine-specific paths.
CHECKPOINT_DIR = os.environ.get("DESIGNLORA_CHECKPOINT_DIR", "./checkpoints")  # per-user LoRA checkpoints
RUNS_DIR = os.environ.get("DESIGNLORA_RUNS_DIR", "./runs/designlora")
LOGS_DIR = os.environ.get("DESIGNLORA_LOGS_DIR", "./designlora_logs")
DEFAULT_CONFIGS = ["configs/base.yaml", "configs/app/app.yaml"]

logging.basicConfig(
    level=os.environ.get("DESIGNLORA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

# Create FastAPI app
app = FastAPI(title="DesignLoRA Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Pydantic Models =============
class DesignLoraRequest(BaseModel):
    userID: str
    selectedIndex: int = Field(ge=0, le=3)
    question: str

class ExperimentSetup(BaseModel):
    userID: str
    imageModel: str = Field(default="schnell", description="schnell, dev, flux2, flux2_klein, or qwen_image")
    prompt: str
    system: str = Field(default="designlora", description="designlora system")
    vlmModel: str = Field(default="idefics2", description="VLM model to use")
    seed: Optional[int] = None

class ImageGenerationRequest(BaseModel):
    userID: str
    prompt: str
    num_images: int = Field(default=4, ge=1, le=10)
    seed: Optional[int] = None
    system: str = Field(default="designlora", description="designlora system")

class CheckpointVersion(BaseModel):
    iteration: int
    checkpoint_path: str
    questions_used: List[str]
    created_at: str
    training_steps: int = 0
    global_step: int = 0
    parent_version: Optional[int] = None  # Track branching: which version this was trained from

class CheckpointInfo(BaseModel):
    checkpoint_id: str
    current_iteration: int
    created_at: str
    last_updated: str
    versions: List[CheckpointVersion]

class FineTuneRequest(BaseModel):
    userID: str
    checkpoint_id: str
    selectedIndex: int = Field(ge=0, le=3)
    question: str
    prompt: Optional[str] = None  # Optional: for auto-setup, if not provided uses existing session prompt
    imageModel: str = Field(default="schnell", description="schnell, dev, flux2, flux2_klein, or qwen_image")
    version: Optional[int] = None  # Optional: specific version to load, defaults to latest
    current_item_keys: Optional[List[str]] = None  # Optional: item_keys of current panel (for image replacement feature)

class GenerateWithLoRARequest(BaseModel):
    userID: str
    checkpoint_id: str
    prompt: str
    num_images: int = Field(default=4, ge=1, le=10)
    seed: Optional[int] = None
    imageModel: str = Field(default="schnell", description="schnell, dev, flux2, flux2_klein, or qwen_image")
    version: Optional[int] = None  # Optional: specific version to load, defaults to latest

# ============= Global Model Cache =============
class ModelCache:
    def __init__(self):
        self.pipe = None
        self.vlm = None
        self.vlm_processor = None
        self.config_hash = None

    def get_config_hash(self, pipe_config, vlm_config):
        """Generate hash for model configuration to detect changes."""
        import hashlib
        config_str = str(sorted(pipe_config.items())) + str(sorted(vlm_config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def load_models_if_needed(self, pipe_config, vlm_config, user_id):
        """Load models only if configuration changed or models not loaded."""
        current_hash = self.get_config_hash(pipe_config, vlm_config)

        if self.config_hash != current_hash:
            # Clear existing models before loading new ones
            self.clear_models()

            logging.info(f"Loading pipe for user {user_id} (config changed)")
            logging.info(f"Pipe config: pipe_id={pipe_config.get('pipe_id')}, pipe_cls={pipe_config.get('pipe_cls')}")

            # scheduler_config를 분리해서 전달
            pipe_config_copy = pipe_config.copy()
            scheduler_config = pipe_config_copy.pop('scheduler_config', {})

            self.pipe = dig_helpers.load_pipe(scheduler_config=scheduler_config, **pipe_config_copy)

            # Log actual pipeline type loaded
            pipe_cls = dig_helpers.get_pipe_cls(self.pipe)
            logging.info(f"Loaded pipeline type: {pipe_cls} (class: {type(self.pipe).__name__})")

            logging.info(f"Loading VLM for user {user_id}")
            self.vlm, self.vlm_processor = dig_helpers.load_vlm(**vlm_config)

            self.config_hash = current_hash
            logging.info(f"Models loaded successfully for user {user_id}")
        else:
            # Log cached pipeline type
            pipe_cls = dig_helpers.get_pipe_cls(self.pipe) if self.pipe else "None"
            logging.info(f"Reusing cached models for user {user_id} (pipe_type: {pipe_cls})")

        return self.pipe, self.vlm, self.vlm_processor

    def clear_models(self):
        """Clear loaded models to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.vlm is not None:
            del self.vlm
            self.vlm = None
        if self.vlm_processor is not None:
            del self.vlm_processor
            self.vlm_processor = None
        torch.cuda.empty_cache()
        self.config_hash = None

model_cache = ModelCache()

# ============= UserSession Class =============
class UserSession:
    def __init__(self, user_id: str, system: str = "designlora", checkpoint_id: Optional[str] = None):
        self.user_id = user_id
        self.system = system
        self.checkpoint_id = checkpoint_id  # NEW: Track which checkpoint is being used
        self.iter = 0
        self.prompt = ""
        self.current_imgs: List[Image.Image] = []
        self.current_seeds: List[int] = []
        self.selected_idx: Optional[int] = None
        self.config = None
        self.optimizer = None
        self.pipe = None
        self.lora_scale = 0.0
        self.best_mu = 0.0
        self.global_step_counter = 0
        self.session_dir = None

        # For BT system
        self.history = []
        self.item_desc: Dict[str, str] = {}
        self.item_img: Dict[str, Image.Image] = {}
        self.bt_scores: Dict[str, float] = {}
        self.seed_to_global: Dict[Tuple[int, str], int] = {}  # (seed, prompt) -> global_idx
        self.obs_order: List[str] = []
        self.current_item_keys: List[str] = []
        self.global_item_counter = 0
        self.current_global_indices: List[int] = []
        self.pref_profile = ""
        self.selection_insights: List[dict] = []
        self.pref_images: List[Image.Image] = []
        self.prev_winner_seed: Optional[int] = None
        self.prev_winner_global_idx: Optional[int] = None
        self.last_ckpt: Optional[str] = None  # Keep for backward compatibility
        self.setup_seed: Optional[int] = None
        self.setup_timestamp: Optional[float] = None

    @property
    def vlm(self):
        """Access shared VLM from global cache."""
        return model_cache.vlm

    @property
    def vlm_processor(self):
        """Access shared VLM processor from global cache."""
        return model_cache.vlm_processor

# Store active sessions
sessions: Dict[str, UserSession] = {}

# ============= Initialize Logger =============
# Initialize global logger with designLoRA_logs directory
_ = get_logger(LOGS_DIR)

# ============= Helper Functions =============
def compile_config(pipe_name: str, vlm_name: str):
    """Compile configuration from pipe and VLM configs."""
    cfg_files = [
        *DEFAULT_CONFIGS,  # Load base configs first
        f"configs/pipe/{pipe_name}.yaml",  # Then override with specific pipe config
        f"configs/vlm/{vlm_name}.yaml"  # Finally apply VLM config
    ]
    return dig_helpers.load_config(cfg_files)

@torch.no_grad()
def generate_images_simple(
    pipe,
    prompt: str,
    num_images: int = 4,
    seed: Optional[int] = None,
    lora_scale: float = 0.0,
    gen_kwargs: dict = None,
    seed_list: Optional[List[int]] = None
) -> Tuple[List[Image.Image], List[int]]:
    """Generate images using the pipeline."""
    if seed_list is not None:
        seeds = seed_list
    elif seed is not None:
        random.seed(seed)
        seeds = [random.randint(0, 99999) for _ in range(num_images)]
    else:
        seeds = [random.randint(0, 99999) for _ in range(num_images)]

    if gen_kwargs is None:
        gen_kwargs = {}

    images = []
    for s in seeds:
        with dig_helpers.LoraManager(pipe, enter_weights=lora_scale):
            generator = torch.Generator().manual_seed(s)
            img = dig_helpers.run_pipe(
                pipe=pipe,
                prompt=[prompt],
                generator_kwargs=gen_kwargs,
                generator=generator,
                num_images_per_prompt=1
            )[0]
            images.append(img)

    return images, seeds

def _register_item(sess, key: str, img: Image.Image, prompt_text: str):
    """Register item in session"""
    if key not in sess.item_img:
        sess.item_img[key] = img
        sess.obs_order.append(key)
    if key not in sess.item_desc:
        desc = describe_image(img, prompt_text)
        sess.item_desc[key] = desc

def _restore_panel_from_keys(session: UserSession, item_keys: List[str]) -> bool:
    """
    Restore current panel from item_keys (for image replacement feature).
    Returns True if successful, False if any images are missing.
    """
    import re

    if not item_keys or len(item_keys) != 4:
        logging.warning(f"Invalid item_keys: expected 4, got {len(item_keys) if item_keys else 0}")
        return False

    restored_images = []
    restored_seeds = []
    restored_keys = []
    restored_global_indices = []

    for key in item_keys:
        # Check if image exists in session
        if key not in session.item_img:
            logging.warning(f"Image not found for key: {key}")
            return False

        # Parse seed and global index from key
        # Format: "panel{iteration}_slot{0-3}_seed{seed}_global{global_idx}"
        seed_match = re.search(r'seed(\d+)', key)
        global_match = re.search(r'global(\d+)', key)

        if not seed_match or not global_match:
            logging.warning(f"Invalid key format: {key}")
            return False

        seed = int(seed_match.group(1))
        global_idx = int(global_match.group(1))

        restored_images.append(session.item_img[key])
        restored_seeds.append(seed)
        restored_keys.append(key)
        restored_global_indices.append(global_idx)

    # Update session's current panel
    session.current_imgs = restored_images
    session.current_seeds = restored_seeds
    session.current_item_keys = restored_keys
    session.current_global_indices = restored_global_indices

    logging.info(f"Restored panel from item_keys:")
    for i, key in enumerate(restored_keys):
        logging.info(f"Slot {i}: {key} (seed={restored_seeds[i]})")

    return True

def _recompute_bt_scores(sess):
    """Recompute BT goodness scores for all observed items from their global indices."""
    import re

    def _global_of(key):
        m = re.search(r'global(\d+)', key)
        return f"global{m.group(1)}" if m else None

    # One BT score per unique global index (0.5 fallback if the surrogate has no estimate yet).
    global_scores = {}
    for k in sess.obs_order:
        g = _global_of(k)
        if g and g not in global_scores:
            try:
                global_scores[g] = float(llm_surrogate.bt_score(g))
            except Exception:
                global_scores[g] = 0.5

    # Assign to every observed item, then propagate to the current panel keys.
    for k in sess.obs_order:
        sess.bt_scores[k] = global_scores.get(_global_of(k), 0.5)
    for key in getattr(sess, 'current_item_keys', []):
        sess.bt_scores[key] = global_scores.get(_global_of(key), 0.5)

def get_or_create_session(user_id: str, system: str = "designlora") -> UserSession:
    """Get existing session or create new one."""
    session_key = f"{user_id}_{system}"
    if session_key not in sessions:
        sessions[session_key] = UserSession(user_id, system)
        _restore_session_state(sessions[session_key])
    return sessions[session_key]

def _restore_session_state(session: UserSession):
    """Restore session state from previous runs."""
    try:
        session_pattern = os.path.join(RUNS_DIR, f"{session.user_id}_{session.system}_*")
        import glob
        existing_sessions = glob.glob(session_pattern)

        if existing_sessions:
            latest_session_dir = max(existing_sessions, key=os.path.getctime)
            session_info_file = os.path.join(latest_session_dir, "session_info.json")

            if os.path.exists(session_info_file):
                with open(session_info_file, 'r') as f:
                    session_data = json.load(f)

                session.iter = session_data.get("iter", 0)
                session.global_step_counter = session_data.get("global_step_counter", 0)
                session.global_item_counter = session_data.get("global_item_counter", 0)
                session.lora_scale = session_data.get("lora_scale", 1.0)
                session.prev_winner_seed = session_data.get("prev_winner_seed")
                session.prev_winner_global_idx = session_data.get("prev_winner_global_idx")
                session.last_ckpt = session_data.get("last_ckpt")
                session.setup_seed = session_data.get("setup_seed")
                session.setup_timestamp = session_data.get("setup_timestamp")
                session.pref_profile = session_data.get("pref_profile", "")
                # Convert string keys back to (seed, prompt) tuples
                seed_to_global_raw = session_data.get("seed_to_global", {})
                session.seed_to_global = {}
                for key_str, value in seed_to_global_raw.items():
                    try:
                        # Key format: "seed_XXXXX_prompt_HASH"
                        if key_str.startswith("(") and key_str.endswith(")"):
                            # Old format: eval the tuple string (unsafe but temporary)
                            import ast
                            seed, prompt = ast.literal_eval(key_str)
                            session.seed_to_global[(seed, prompt)] = value
                        else:
                            # Just ignore old format
                            pass
                    except:
                        pass

                logging.info(f"Restored session state for user {session.user_id}: iter={session.iter}, global_step={session.global_step_counter}")

    except Exception as e:
        logging.warning(f"Failed to restore session state for user {session.user_id}: {e}")

def _save_session_state(session: UserSession):
    """Save session state for future restoration."""
    if not session.session_dir:
        return

    try:
        # Convert (seed, prompt) tuples to strings for JSON serialization
        seed_to_global_serializable = {
            str(key): value for key, value in session.seed_to_global.items()
        }

        session_info = {
            "user_id": session.user_id,
            "system": session.system,
            "iter": session.iter,
            "global_step_counter": session.global_step_counter,
            "global_item_counter": session.global_item_counter,
            "lora_scale": session.lora_scale,
            "prev_winner_seed": session.prev_winner_seed,
            "prev_winner_global_idx": session.prev_winner_global_idx,
            "last_ckpt": session.last_ckpt,
            "setup_seed": session.setup_seed,
            "setup_timestamp": session.setup_timestamp,
            "pref_profile": session.pref_profile,
            "seed_to_global": seed_to_global_serializable,
            "saved_at": time.time()
        }

        session_info_file = session.session_dir / "session_info.json"
        with open(session_info_file, 'w') as f:
            json.dump(session_info, f, indent=2)

        logging.debug(f"Saved session state for user {session.user_id}")

    except Exception as e:
        logging.warning(f"Failed to save session state for user {session.user_id}: {e}")

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def _save_checkpoint(user_id: str, checkpoint_id: str, pipe, iteration: int, question: str, metadata: dict = None) -> str:
    """Save LoRA checkpoint with version history."""
    try:
        # Create checkpoint directory structure: checkpoints/{userID}/{checkpoint_id}/
        checkpoint_dir = Path(CHECKPOINT_DIR) / user_id / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        ckpt_base = checkpoint_dir / f"lora_iter_{iteration:03d}"
        dig_helpers.save_weights(pipe, str(ckpt_base))
        checkpoint_path = f"{ckpt_base}.pt"

        # Load or create metadata
        metadata_file = checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                checkpoint_metadata = json.load(f)
        else:
            checkpoint_metadata = {
                "checkpoint_id": checkpoint_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "current_iteration": 0,
                "versions": []
            }

        # Create version entry
        version_entry = {
            "iteration": iteration,
            "checkpoint_path": checkpoint_path,
            "questions_used": [question] if question else [],
            "created_at": datetime.now().isoformat(),
            "training_steps": metadata.get("training_steps", 0) if metadata else 0,
            "global_step": metadata.get("global_step", 0) if metadata else 0,
            "parent_version": metadata.get("parent_version") if metadata else None  # Track branching
        }

        # Update or add version
        existing_version = None
        for i, v in enumerate(checkpoint_metadata.get("versions", [])):
            if v["iteration"] == iteration:
                existing_version = i
                break

        if existing_version is not None:
            # Update existing version (merge questions)
            existing_questions = checkpoint_metadata["versions"][existing_version].get("questions_used", [])
            if question and question not in existing_questions:
                existing_questions.append(question)
            checkpoint_metadata["versions"][existing_version] = version_entry
            checkpoint_metadata["versions"][existing_version]["questions_used"] = existing_questions
        else:
            # Add new version
            checkpoint_metadata.setdefault("versions", []).append(version_entry)

        # Update top-level metadata
        checkpoint_metadata["current_iteration"] = iteration
        checkpoint_metadata["last_updated"] = datetime.now().isoformat()

        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)

        logging.info(f"Saved checkpoint: {checkpoint_id}, iteration: {iteration}, path: {checkpoint_path}")
        return checkpoint_path

    except Exception as e:
        logging.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
        raise

def _load_checkpoint(user_id: str, checkpoint_id: str, pipe, version: Optional[int] = None) -> dict:
    """Load LoRA checkpoint and return metadata. If version is specified, loads that version."""
    try:
        checkpoint_dir = Path(CHECKPOINT_DIR) / user_id / checkpoint_id
        metadata_file = checkpoint_dir / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found for user {user_id}")

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Find the version to load
        versions = metadata.get("versions", [])
        if not versions:
            raise FileNotFoundError(f"No versions found for checkpoint {checkpoint_id}")

        if version is not None:
            # Load specific version
            target_version = None
            for v in versions:
                if v["iteration"] == version:
                    target_version = v
                    break
            if target_version is None:
                raise FileNotFoundError(f"Version {version} not found for checkpoint {checkpoint_id}")
        else:
            # Load latest version
            target_version = max(versions, key=lambda x: x["iteration"])

        # Load LoRA weights
        checkpoint_path = target_version["checkpoint_path"]
        if checkpoint_path and os.path.exists(checkpoint_path):
            base_path = checkpoint_path.replace('.pt', '')
            dig_helpers.load_weights(pipe, base_path)
            logging.info(f"Loaded checkpoint: {checkpoint_id}, version: {target_version['iteration']}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Return metadata with the loaded version info
        return {
            **metadata,
            "loaded_version": target_version,
            "iteration": target_version["iteration"]  # For backward compatibility
        }

    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
        raise

def _list_checkpoints(user_id: str) -> List[CheckpointInfo]:
    """List all checkpoints with all versions for a user."""
    try:
        user_checkpoint_dir = Path(CHECKPOINT_DIR) / user_id

        if not user_checkpoint_dir.exists():
            return []

        checkpoints = []
        for checkpoint_dir in user_checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir():
                metadata_file = checkpoint_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Convert versions to CheckpointVersion objects
                    versions = []
                    for v in metadata.get("versions", []):
                        versions.append(CheckpointVersion(
                            iteration=v.get("iteration", 0),
                            checkpoint_path=v.get("checkpoint_path", ""),
                            questions_used=v.get("questions_used", []),
                            created_at=v.get("created_at", ""),
                            training_steps=v.get("training_steps", 0),
                            global_step=v.get("global_step", 0),
                            parent_version=v.get("parent_version")
                        ))

                    checkpoints.append(CheckpointInfo(
                        checkpoint_id=metadata.get("checkpoint_id", checkpoint_dir.name),
                        current_iteration=metadata.get("current_iteration", 0),
                        created_at=metadata.get("created_at", ""),
                        last_updated=metadata.get("last_updated", ""),
                        versions=versions
                    ))

        # Sort by last updated (most recent first)
        checkpoints.sort(key=lambda x: x.last_updated, reverse=True)
        return checkpoints

    except Exception as e:
        logging.error(f"Failed to list checkpoints for user {user_id}: {e}")
        return []

def _auto_setup_if_needed(session: UserSession, user_id: str, prompt: str = None, image_model: str = "schnell", system: str = "designlora"):
    """
    Auto-setup session if models are not loaded.
    This allows clients to skip the /setup call and directly use /finetune or /generate_with_lora.
    """
    # Check if session needs setup
    needs_setup = (session.pipe is None or session.config is None)

    if not needs_setup:
        logging.info(f"Session already initialized for user {user_id}")
        return

    logging.info(f"Auto-setup: Initializing session for user {user_id}")

    # Map image model to pipe config
    pipe_map = {
        "schnell": "schnell",
        "dev": "dev",
        "qwen_image": "qwen_image",
        "flux2": "flux2",
        "flux2_klein": "flux2_klein",
    }
    pipe_name = pipe_map.get(image_model, "schnell")

    # Load configuration
    vlm_name = "idefics2"
    session.config = compile_config(pipe_name, vlm_name)

    # Load models using global cache
    model_cache.load_models_if_needed(
        session.config["pipe_kwargs"],
        session.config["vlm_kwargs"],
        user_id
    )

    # Initialize session.pipe
    session.pipe = model_cache.pipe

    # Setup LoRA for 'designlora' system
    if session.system == "designlora":
        # Only create new LoRA and optimizer if they don't exist
        if not hasattr(session, 'optimizer') or session.optimizer is None:
            lora_handle = dig_helpers.create_lora(model_cache.pipe, **session.config.get("lora_kwargs", {}))
            lora_params = lora_handle["params"] if isinstance(lora_handle, dict) else lora_handle
            session.optimizer = torch.optim.Adam(lora_params, lr=session.config.get("lora_kwargs", {}).get("lora_lr", 5e-5))
            logging.info(f"Created new LoRA and optimizer for user {user_id}")
        else:
            logging.info(f"Reusing existing LoRA and optimizer for user {user_id}")

    # Setup VLM context
    edit_base = {
        "vlm": session.vlm,
        "vlm_processor": session.vlm_processor,
        "cfg": session.config
    }
    setup_vlm_context(session.pipe, edit_base)

    # Initialize session tracking
    if not hasattr(session, 'setup_seed') or session.setup_seed is None:
        session.setup_seed = random.randint(0, 99999)
        session.setup_timestamp = time.time()

    # Set prompt if provided
    if prompt:
        session.prompt = prompt

    # Create session directory for logging
    api_logger = get_logger()
    setup_data = {
        "userID": user_id,
        "system": system,
        "imageModel": image_model,
        "vlmModel": vlm_name,
        "prompt": prompt or session.prompt,
        "seed": session.setup_seed,
        "pipe_name": pipe_name,
        "vlm_name": vlm_name,
        "setup_timestamp": session.setup_timestamp,
        "auto_setup": True
    }

    if not session.session_dir:
        session.session_dir = api_logger.log_setup(f"{user_id}_{system}", setup_data)

    # Save session state after setup
    _save_session_state(session)

    # Clear Bradley-Terry state for designlora system
    if session.system == "designlora":
        llm_surrogate.bt_clear()
        logging.info(f"Cleared BT state for {session.system} system")

    logging.info(f"Auto-setup completed for user {user_id}")

# ============= API Endpoints =============
@app.post("/setup")
async def setup_experiment(request: ExperimentSetup):
    """Initialize experiment with user settings."""
    try:
        # Get logger
        api_logger = get_logger()

        # Clear models and force reload on every /setup call
        model_cache.clear_models()
        model_cache.config_hash = None

        session = get_or_create_session(request.userID, request.system)
        session.prompt = request.prompt

        # Map image model to pipe config
        pipe_map = {
            "schnell": "schnell",
            "dev": "dev",
            "qwen_image": "qwen_image",
            "flux2": "flux2",
            "flux2_klein": "flux2_klein",
        }
        pipe_name = pipe_map.get(request.imageModel, "schnell")

        # Load configuration
        vlm_name = "idefics2"
        session.config = compile_config(pipe_name, vlm_name)

        # Load models using global cache
        model_cache.load_models_if_needed(
            session.config["pipe_kwargs"],
            session.config["vlm_kwargs"],
            request.userID
        )

        # Initialize session.pipe
        session.pipe = model_cache.pipe

        # Setup LoRA for 'designlora' system
        if session.system == "designlora":
            # Only create new LoRA and optimizer if they don't exist
            if not hasattr(session, 'optimizer') or session.optimizer is None:
                lora_handle = dig_helpers.create_lora(model_cache.pipe, **session.config.get("lora_kwargs", {}))
                lora_params = lora_handle["params"] if isinstance(lora_handle, dict) else lora_handle
                session.optimizer = torch.optim.Adam(lora_params, lr=session.config.get("lora_kwargs", {}).get("lora_lr", 5e-5))
                logging.info(f"Created new LoRA and optimizer for user {session.user_id}")
            else:
                logging.info(f"Reusing existing LoRA and optimizer for user {session.user_id}")

            # Load previous LoRA if exists for this user
            log_msg = f" LoRA Setup for user {session.user_id}:"
            logging.info(log_msg)

            if session.last_ckpt and os.path.exists(session.last_ckpt):
                try:
                    logging.info(f"Loading stored LoRA checkpoint: {session.last_ckpt}")
                    dig_helpers.load_weights(session.pipe, session.last_ckpt.replace('.pt', ''))
                    logging.info(f"Successfully loaded LoRA from: {session.last_ckpt}")
                except Exception as e:
                    logging.warning(f"Failed to load previous LoRA: {e}")
                    session.last_ckpt = None
            else:
                # Check for existing checkpoints in logs directory
                user_pattern = f"{request.userID}_{session.system}_*"
                sessions_dir = Path(f"{LOGS_DIR}/sessions")
                if sessions_dir.exists():
                    import glob
                    user_dirs = glob.glob(str(sessions_dir / user_pattern))
                    if user_dirs:
                        latest_user_dir = max(user_dirs, key=os.path.getctime)
                        ckpt_pattern = os.path.join(latest_user_dir, "interactions", "*", "lora_after_iter_*.pt")
                        existing_ckpts = glob.glob(ckpt_pattern)
                        search_msg = f"   Searching for LoRA in: {ckpt_pattern}"
                        found_msg = f"   Found {len(existing_ckpts)} existing checkpoints"
                        logging.info(search_msg)
                        logging.info(found_msg)

                        if existing_ckpts:
                            latest_ckpt = max(existing_ckpts, key=os.path.getctime)
                            try:
                                logging.info(f"Found existing LoRA checkpoint: {latest_ckpt}")
                                dig_helpers.load_weights(session.pipe, latest_ckpt.replace('.pt', ''))
                                session.last_ckpt = latest_ckpt
                                logging.info(f"Successfully loaded existing LoRA from: {latest_ckpt}")
                            except Exception as e:
                                logging.warning(f"Failed to load existing LoRA: {e}")
                        else:
                            logging.info(f"No existing LoRA found - starting fresh")
                    else:
                        logging.info(f"No previous user sessions found - starting fresh")
                else:
                    logging.info(f"No {LOGS_DIR}/sessions directory - starting fresh")

        # Setup VLM context
        edit_base = {
            "vlm": session.vlm,
            "vlm_processor": session.vlm_processor,
            "cfg": session.config
        }
        setup_vlm_context(session.pipe, edit_base)

        # Initialize session tracking
        if not hasattr(session, 'setup_seed') or session.setup_seed is None:
            session.setup_seed = request.seed if request.seed is not None else random.randint(0, 99999)
            session.setup_timestamp = time.time()

        # Log setup
        setup_data = {
            "userID": request.userID,
            "system": request.system,
            "imageModel": request.imageModel,
            "vlmModel": request.vlmModel,
            "prompt": request.prompt,
            "seed": session.setup_seed,
            "pipe_name": pipe_name,
            "vlm_name": vlm_name,
            "setup_timestamp": session.setup_timestamp
        }
        session.session_dir = api_logger.log_setup(f"{request.userID}_{session.system}", setup_data)

        # Save session state after setup
        _save_session_state(session)

        # Clear Bradley-Terry state for designlora system
        if session.system == "designlora":
            llm_surrogate.bt_clear()
            logging.info(f"Cleared BT state for {session.system} system")

        # Log successful setup
        api_logger.log_system_event(
            event_type="setup_complete",
            user_id=request.userID,
            data=setup_data
        )

        return JSONResponse({
            "status": "success",
            "message": f"Experiment setup complete for {request.system} system",
            "userID": request.userID,
            "system": session.system,
            "imageModel": request.imageModel,
            "vlmModel": request.vlmModel
        })

    except Exception as e:
        logging.error(f"Setup failed: {str(e)}")
        api_logger = get_logger()
        api_logger.log_system_event(
            event_type="error",
            user_id=request.userID,
            data={"endpoint": "/setup", "request": request.dict()},
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_images(request: ImageGenerationRequest):
    """Generate images based on prompt."""
    try:
        # Get logger
        api_logger = get_logger()

        session = get_or_create_session(request.userID, request.system)

        if session.pipe is None:
            raise HTTPException(status_code=400, detail="Please call /setup first")

        if session.config is None:
            raise HTTPException(status_code=400, detail="Session config not initialized. Please call /setup first")

        # Generate images
        gen_kwargs = session.config.get("generator_kwargs", {})

        # /generate endpoint always uses base model (no LoRA)
        base_lora_scale = 0.0
        logging.info(f"Generating with base model (LoRA scale: {base_lora_scale})")

        # Generate 4 seeds based on base_seed
        if request.num_images == 4:
            if request.seed is not None:
                # Client provided base seed: create variants
                base_seed = request.seed
                fixed_seeds = [base_seed, base_seed + 10, base_seed + 20, base_seed + 30]
                logging.info(f"Using base seed {base_seed}  seeds: {fixed_seeds}")
            else:
                # No seed provided: use default
                fixed_seeds = [10, 20, 30, 40]
                logging.info(f"Using default seeds: {fixed_seeds}")

            images, seeds = generate_images_simple(
                pipe=session.pipe,
                prompt=request.prompt,
                num_images=request.num_images,
                seed=None,
                lora_scale=base_lora_scale,
                gen_kwargs=gen_kwargs,
                seed_list=fixed_seeds
            )
        else:
            # For non-4 images, use provided seed or random
            images, seeds = generate_images_simple(
                pipe=session.pipe,
                prompt=request.prompt,
                num_images=request.num_images,
                seed=request.seed,
                lora_scale=base_lora_scale,
                gen_kwargs=gen_kwargs
            )

        # Store current images and seeds
        session.current_imgs = images
        session.current_seeds = seeds

        # Register items with global indices
        session.current_item_keys = []
        session.current_global_indices = []

        for i, (seed, img) in enumerate(zip(seeds, images)):
            # Create global index for new (seed, prompt) combinations
            seed_prompt_key = (seed, request.prompt)
            if seed_prompt_key in session.seed_to_global:
                global_idx = session.seed_to_global[seed_prompt_key]
            else:
                global_idx = session.global_item_counter
                session.global_item_counter += 1
                session.seed_to_global[seed_prompt_key] = global_idx

            session.current_global_indices.append(global_idx)
            key = f"panel{session.iter:02d}_slot{i}_seed{seed}_global{global_idx:03d}"
            session.current_item_keys.append(key)
            _register_item(session, key, img, request.prompt)
            session.bt_scores[key] = 0.5

        logging.info(f"Registered {len(session.current_item_keys)} images with item_keys: {session.current_item_keys}")

        # Convert images to base64
        image_data = []
        for img in images:
            image_data.append(image_to_base64(img))

        # Save generated images to panel-specific directory
        if session.session_dir:
            try:
                # Create panel-specific directory
                generated_dir = session.session_dir / "generated_images" / f"panel_{session.iter:02d}"
                generated_dir.mkdir(parents=True, exist_ok=True)

                # Save images with seed in filename
                saved_paths = []
                for seed, img in zip(seeds, images):
                    img_path = generated_dir / f"image_seed_{seed:05d}.png"
                    img.save(img_path)
                    saved_paths.append(str(img_path))

                # Save metadata
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "iteration": session.iter,
                    "system": session.system,
                    "prompt": request.prompt,
                    "num_images": request.num_images,
                    "seeds": seeds,
                    "item_keys": session.current_item_keys,
                    "stage": "generation"
                }
                metadata_path = generated_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                logging.info(f"Generated images saved to: {generated_dir}")
                logging.info(f"{len(saved_paths)} images saved")
            except Exception as e:
                logging.warning(f"Failed to save generated images: {e}")

        # Log generation
        api_logger.log_system_event(
            event_type="image_generation",
            user_id=request.userID,
            data={
                "prompt": request.prompt,
                "num_images": request.num_images,
                "seeds": seeds,
                "iteration": session.iter,
                "item_keys": session.current_item_keys
            }
        )

        logging.info(f"/generate COMPLETED - Returning {len(session.current_item_keys)} item_keys")
        logging.info(f"item_keys: {session.current_item_keys}")

        return JSONResponse({
            "status": "success",
            "images": image_data,
            "seeds": seeds,
            "item_keys": session.current_item_keys,  # For History feature
            "prompt": request.prompt,
            "iteration": session.iter
        })

    except Exception as e:
        logging.error(f"Image generation failed: {str(e)}")
        import traceback
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        api_logger = get_logger()
        api_logger.log_system_event(
            event_type="error",
            user_id=request.userID,
            data={"endpoint": "/generate", "request": request.dict()},
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interact/designlora")
async def interact_designlora(request: DesignLoraRequest):
    """Handle interaction for 'designlora' system (BT + EI + Question logit)."""
    try:
        # Get logger
        api_logger = get_logger()

        session = get_or_create_session(request.userID, "designlora")

        logging.info(
            "Interaction: user=%s iter=%s observed=%s panel=%s ckpt=%s question=%r",
            session.user_id, session.iter, len(session.obs_order),
            len(session.current_item_keys), session.last_ckpt or "None", request.question,
        )


        if session.system != "designlora":
            raise HTTPException(status_code=400, detail="Session not configured for the 'designlora' system")

        if not session.current_imgs:
            raise HTTPException(status_code=400, detail="No images to select from")

        sel_idx = request.selectedIndex
        if sel_idx < 0 or sel_idx >= len(session.current_imgs):
            raise HTTPException(status_code=400, detail="Invalid selection index")

        # Get train_weight from config (default: 5, where alpha = weight * rank)
        train_weight = session.config.get("opt_kwargs", {}).get("train_weight", 5)

        # Load previous LoRA checkpoint if exists (for cumulative training)
        if session.last_ckpt and os.path.exists(session.last_ckpt):
            try:
                base_path = session.last_ckpt.replace('.pt', '')
                dig_helpers.load_weights(session.pipe, base_path)
                session.lora_scale = train_weight  # Use train_weight from config
                load_msg = f"   Loaded previous LoRA for cumulative training: {session.last_ckpt} (train_weight={train_weight})"
                logging.info(load_msg)
            except Exception as e:
                logging.warning(f"Failed to load previous LoRA: {e}")
                session.lora_scale = train_weight  # Still use train_weight for fresh LoRA training
        else:
            # First interaction - no previous LoRA, but still train with train_weight
            session.lora_scale = train_weight
            first_msg = f"   First interaction - training new LoRA (train_weight={train_weight})"
            logging.info(first_msg)


        # Update selection
        session.selected_idx = sel_idx
        winner_img = session.current_imgs[sel_idx]
        winner_seed = session.current_seeds[sel_idx]

        # Use the same key format as the current panel
        winner_key = session.current_item_keys[sel_idx]

        # Use existing panel keys for losers
        loser_keys = []
        all_keys = []
        for i, img in enumerate(session.current_imgs):
            if i == sel_idx:
                all_keys.append(winner_key)
            else:
                loser_key = session.current_item_keys[i]
                loser_keys.append(loser_key)
                all_keys.append(loser_key)

        # BT batch update with all preference pairs (winner > all losers)
        if loser_keys:
            import re
            winner_global = None
            loser_globals = []

            # Extract winner's global index
            winner_match = re.search(r'global(\d+)', winner_key)
            if winner_match:
                winner_global = f"global{winner_match.group(1)}"

            # Extract losers' global indices
            for lk in loser_keys:
                loser_match = re.search(r'global(\d+)', lk)
                if loser_match:
                    loser_globals.append(f"global{loser_match.group(1)}")

            # Create preference pairs using global indices
            if winner_global and loser_globals:
                preference_pairs = [(winner_global, lg) for lg in loser_globals]
                bt_update_msg = f" BT Update: {len(preference_pairs)} preference pairs using global indices"
                logging.info(bt_update_msg)

                for winner, loser in preference_pairs:
                    pair_msg = f"   {winner} > {loser}"
                    logging.info(pair_msg)
                try:
                    llm_surrogate.bt_update(preference_pairs)
                    success_msg = f"   BT update successful"
                    logging.info(success_msg)
                except Exception as e:
                    logging.error(f"BT update failed: {e}")

        # Recompute BT scores for ALL observed items
        logging.info(f"Recomputing BT scores for ALL {len(session.obs_order)} accumulated items")
        old_scores = session.bt_scores.copy()
        _recompute_bt_scores(session)

        # === DESIGNLORA: 20-Candidate evaluation with Question + Preference logits ===

        if session.optimizer and session.config:
            try:
                # Get recent evaluation data - use BT scores as preference baseline
                # IMPORTANT: VLM uses ONLY the current panel's 4 images as reference (not accumulated history)
                # This is a design choice - while BT scores are accumulated globally across all iterations,
                # the VLM evaluation only sees the most recent 4 images to keep the prompt manageable.
                recent_eval_data = []
                if hasattr(session, 'current_item_keys') and hasattr(session, 'current_seeds') and session.current_item_keys:
                    for i, key in enumerate(session.current_item_keys):
                        if key in session.item_img and key in session.bt_scores and i < len(session.current_seeds):
                            recent_eval_data.append({
                                'key': key,
                                'image': session.item_img[key],
                                'score': session.bt_scores[key],
                                'seed': session.current_seeds[i]
                            })

                if not recent_eval_data:
                    return {"status": "success", "message": "No reference data available"}

                # Start 20-candidate evaluation and training
                step1_start_time = time.time()
                N_CANDIDATES_PER_CYCLE = 20
                total_training_steps = 0
                # Merge train_generator_kwargs with generator_kwargs (train overrides gen)
                gen_kwargs_base = session.config.get("generator_kwargs", {})
                train_kwargs = {**gen_kwargs_base, **session.config.get("train_generator_kwargs", {})}

                # === Prepare reference data once (shared by all candidates) ===
                ref_bt_scores = [ref_data['score'] for ref_data in recent_eval_data[:4]]

                # Build prompt with A={score} B={score} format
                score_labels = []
                letters = ['A', 'B', 'C', 'D']
                for j, ref_data in enumerate(recent_eval_data[:4]):
                    letter = letters[j]
                    score = ref_data['score'] * 100
                    score_labels.append(f"{letter}={score:.1f}%")

                score_info = " ".join(score_labels)

                # Create pairwise comparisons based on BT scores
                scores = [recent_eval_data[i]['score'] for i in range(len(recent_eval_data[:4]))]
                sorted_indices = sorted(range(len(recent_eval_data[:4])), key=lambda i: recent_eval_data[i]['score'], reverse=True)

                # Build ranking with equal scores shown as '='
                ranking_parts = []
                i = 0
                while i < len(sorted_indices):
                    current_score = scores[sorted_indices[i]]
                    equal_indices = [sorted_indices[i]]

                    # Find all indices with the same score
                    j = i + 1
                    while j < len(sorted_indices) and abs(scores[sorted_indices[j]] - current_score) < 0.001:
                        equal_indices.append(sorted_indices[j])
                        j += 1

                    # Add to ranking
                    if len(equal_indices) == 1:
                        ranking_parts.append(letters[equal_indices[0]])
                    else:
                        equal_letters = [letters[idx] for idx in equal_indices]
                        ranking_parts.append("= ".join(equal_letters))

                    i = j

                ranking = " > ".join(ranking_parts)

                # Generate pairwise comparisons
                pairwise_comps = []
                for i in range(len(sorted_indices)):
                    for j in range(i+1, len(sorted_indices)):
                        winner_idx = sorted_indices[i]
                        loser_idx = sorted_indices[j]
                        winner_score = scores[winner_idx]
                        loser_score = scores[loser_idx]

                        # Only add if scores are significantly different
                        if abs(winner_score - loser_score) >= 0.001:
                            pairwise_comps.append(f"{letters[winner_idx]} over {letters[loser_idx]}")

                pairwise_str = ", ".join(pairwise_comps) if pairwise_comps else "No significant differences"

                # Create enhanced reference images with labels
                ref_images_labeled = []
                letters = ['A', 'B', 'C', 'D']

                # Identify the actual winner based on user selection
                actual_winner_key = None
                if hasattr(session, 'selected_idx') and session.selected_idx is not None:
                    if hasattr(session, 'current_item_keys') and session.selected_idx < len(session.current_item_keys):
                        actual_winner_key = session.current_item_keys[session.selected_idx]

                for j, ref_data in enumerate(recent_eval_data[:4]):
                    try:
                        from image_utils import add_minimal_label_for_vlm
                        letter_label = letters[j]
                        is_winner = (ref_data['key'] == actual_winner_key) if actual_winner_key else False

                        # Minimal label for VLM input
                        img_for_vlm = add_minimal_label_for_vlm(ref_data['image'], letter_label, is_winner=is_winner)
                        ref_images_labeled.append(img_for_vlm)
                    except Exception as e:
                        ref_images_labeled.append(ref_data['image'])

                # Create VLM prompt combining preference + user question
                combined_qtxt = (
                    "You will see five images in sequence. First four are reference images A, B, C, D. "
                    "The fifth image is the candidate for evaluation.\n"
                    f"Reference preference scores: {score_info} (higher=better).\n"
                    f"Ranking (bestworst): {ranking}.\n"
                    f"Pairwise preferences: {pairwise_str}.\n"
                    "Given the user's preference examples A–D (with WINNER/LOSER labels), would the user prefer the candidate (5th image)? Answer Yes or No.\n"
                    "Assume that the user is likely to prefer images similar to WINNER examples, and unlikely to prefer images similar to OTHER examples. "
                    "If the candidate is similar to both WINNER and OTHER examples, infer the preference based on subtle differences."
                    "would the user prefer the candidate (5th image)? Answer Yes or No."
                )

                # Find the actual winner image based on user selection
                winner_image = None
                if hasattr(session, 'selected_idx') and session.selected_idx is not None:
                    if hasattr(session, 'current_imgs') and session.selected_idx < len(session.current_imgs):
                        winner_image = session.current_imgs[session.selected_idx]

                # Generate candidate seeds
                candidate_seeds = [random.randint(0, 99999) for _ in range(N_CANDIDATES_PER_CYCLE)]

                for seed_idx, fixed_seed in enumerate(candidate_seeds):
                    try:

                        # Create dual-question edit for loss_vlm_multiqa approach
                        qa_dual = [
                            {
                                "question": f"Does the candidate image better reflect the user's question '{request.question}' compared to the reference image? Answer Yes or No.",
                                "answer": "Yes",
                                "ref_images": [winner_image] if winner_image is not None else [],
                                "image_dim": (384, 384)
                            },
                            {
                                "question": combined_qtxt,
                                "answer": "Yes",
                                "ref_images": ref_images_labeled,
                                "image_dim": (384, 384)
                            }
                        ]

                        # Create single edit with both questions
                        edit_dual = dig_pipeline.create_edit(
                            session.pipe, session.vlm, session.vlm_processor,
                            session.config, qa_dual, session.prompt
                        )

                        # Get latent shape and create noise
                        generator = torch.Generator(device=session.pipe.device).manual_seed(fixed_seed)
                        pipe_cls = dig_helpers.get_pipe_cls(session.pipe)
                        if pipe_cls == "flux":
                            latent_shape = dig_helpers.get_flux_latent_shape(session.pipe, train_kwargs, pack=True)
                        elif pipe_cls == "flux2":
                            # FLUX2 expects unpacked 4D latents as input (it packs internally)
                            latent_shape = dig_helpers.get_flux2_latent_shape(session.pipe, train_kwargs, pack=False)
                        elif pipe_cls == "qwen":
                            latent_shape = dig_helpers.get_qwen_latent_shape(session.pipe, train_kwargs, pack=True)
                        else:
                            latent_shape = (1, session.pipe.unet.config.in_channels,
                                          train_kwargs.get("height", 384)//8,
                                          train_kwargs.get("width", 384)//8)

                        init_noise = torch.randn(latent_shape, generator=generator,
                                                device=session.pipe.device, dtype=session.pipe.dtype)

                        # Clear GPU memory before training
                        torch.cuda.empty_cache()

                        # Single inner_loop call
                        base_loss = dig_pipeline.inner_loop(
                            session.pipe, edit_dual, train_kwargs, generator,
                            session.lora_scale, init_noise, total_training_steps
                        )

                        loss = base_loss

                        # Optimizer step
                        session.optimizer.zero_grad()
                        loss.backward()
                        session.optimizer.step()
                        session.global_step_counter += 1

                        step_loss = float(loss)

                        total_training_steps += 1

                    except Exception as e:
                        continue

                step1_end_time = time.time()
                step1_duration = step1_end_time - step1_start_time
                step1_msg = f"   Step 1 (LoRA Finetuning) completed: {total_training_steps} steps in {step1_duration:.2f}s ({step1_duration/total_training_steps:.2f}s per step)"
                logging.info(step1_msg)

                # Update LoRA scale
                session.lora_scale = 1.0

                # Save LoRA checkpoint after training
                if session.session_dir:
                    try:
                        iter_dir = session.session_dir / "interactions" / f"iter_{session.iter:03d}"
                        iter_dir.mkdir(parents=True, exist_ok=True)
                        ckpt_base = iter_dir / f"lora_after_iter_{session.iter+1:02d}"

                        dig_helpers.save_weights(session.pipe, str(ckpt_base))
                        session.last_ckpt = f"{ckpt_base}.pt"
                        logging.info(f"LoRA checkpoint saved successfully: {session.last_ckpt}")
                    except Exception as e:
                        logging.error(f"Failed to save LoRA checkpoint: {str(e)}")

                # === Post-Training: Generate new 4 images with trained LoRA ===
                step2_start_time = time.time()
                base_prompt = session.prompt
                N_CAND = 8
                N_SHOW = 4

                # Clear memory
                torch.cuda.empty_cache()

                # Generate candidates
                cand_seeds = [random.randint(0, 99999) for _ in range(N_CAND)]
                stats = []


                # Reuse reference data from Step 1 (score_info, ranking, pairwise_str, ref_images_labeled, combined_qtxt, winner_image)

                # Generate and evaluate new candidates
                for s in cand_seeds:
                    try:
                        # Generate candidate with trained LoRA
                        gen_kwargs = session.config.get("generator_kwargs", {})
                        with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                            generator = torch.Generator().manual_seed(s)
                            cand_img = dig_helpers.run_pipe(
                                pipe=session.pipe,
                                prompt=[base_prompt],
                                generator_kwargs=gen_kwargs,
                                generator=generator,
                                num_images_per_prompt=1
                            )[0]

                        # Evaluate candidate with VLM (reuse Step 1 structure, only candidate_image differs)
                        qa_dual_eval = [
                            {
                                "question": f"Does the candidate image better reflect the user's question '{request.question}' compared to the reference image? Answer Yes or No.",
                                "answer": "Yes",
                                "ref_images": [winner_image] if winner_image is not None else [],
                                "candidate_image": cand_img,
                                "image_dim": (384, 384)
                            },
                            {
                                "question": combined_qtxt,
                                "answer": "Yes",
                                "ref_images": ref_images_labeled,
                                "candidate_image": cand_img,
                                "image_dim": (384, 384)
                            }
                        ]

                        edit_dual_eval = dig_pipeline.create_edit(
                            session.pipe, session.vlm, session.vlm_processor,
                            session.config, qa_dual_eval, base_prompt
                        )

                        # Run multiple VLM evaluations
                        N_EVAL = 3
                        yes_probs = []

                        for eval_idx in range(N_EVAL):
                            try:
                                eval_img = cand_img

                                loss_eval, meta_eval = dig_operators.loss_vlm_multiqa(
                                    pipe=session.pipe,
                                    edit=edit_dual_eval,
                                    generator_kwargs=gen_kwargs,
                                    pred_x0=eval_img
                                )

                                question_prob = float(meta_eval["probs"][0].cpu().item())
                                combined_with_pref_prob = float(meta_eval["probs"][1].cpu().item())

                                question_weight = session.config.get("designlora_question_weight", 0.5)
                                pref_weight = session.config.get("designlora_pref_weight", 0.5)
                                eval_prob_yes = (question_weight * question_prob + pref_weight * combined_with_pref_prob)
                                yes_probs.append(eval_prob_yes)

                            except Exception as e:
                                logging.warning(f"DesignLora VLM evaluation {eval_idx+1} failed: {e}")
                                yes_probs.append(0.5)

                        # Calculate statistics
                        if yes_probs:
                            mu = np.mean(yes_probs)
                            sigma = np.std(yes_probs) + 0.01
                            prob_yes = mu
                        else:
                            mu, sigma = 0.5, 0.1
                            prob_yes = 0.5

                        # Convert to centered logit
                        import math
                        if 0.001 < prob_yes < 0.999:
                            yes_logit = math.log(prob_yes / (1 - prob_yes))
                        else:
                            yes_logit = 0.0
                        stats.append((s, mu, sigma, cand_img, prob_yes, yes_logit))

                    except Exception as e:
                        stats.append((s, 0.5, 0.1, None, 0.5, 0.0))

                # Expected Improvement scoring
                def expected_improvement(mu, sigma, best=0.5, xi=0.0):
                    from scipy.stats import norm
                    if sigma == 0:
                        return 0.0
                    z = (mu - best - xi) / sigma
                    ei = (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                    return max(0.0, ei)

                # Score and sort candidates
                scored = []
                for stat in stats:
                    seed, mu, sigma, img, token_prob, logit = stat
                    ei_score = expected_improvement(mu, sigma, best=0.5, xi=0.0)
                    scored.append((seed, ei_score, mu, sigma, img, token_prob, logit))

                sorted_scored = sorted(scored, key=lambda x: -x[1])
                top3 = [item[0] for item in sorted_scored[:N_SHOW-1]]

                # Build next panel: winner + top 3 EI candidates
                next_seeds = []
                next_images = []

                # Slot 0: Previous winner with updated LoRA
                seed0 = winner_seed
                next_seeds.append(seed0)

                with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                    generator = torch.Generator().manual_seed(seed0)
                    winner_img_new = dig_helpers.run_pipe(
                        pipe=session.pipe,
                        prompt=[base_prompt],
                        generator_kwargs=gen_kwargs,
                        generator=generator,
                        num_images_per_prompt=1
                    )[0]
                next_images.append(winner_img_new)

                # Slots 1-3: Top EI candidates
                for item in sorted_scored[:N_SHOW-1]:
                    seed, ei_score, mu, sigma, img, token_prob, logit = item
                    next_seeds.append(seed)
                    if img is not None:
                        next_images.append(img)
                    else:
                        with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                            generator = torch.Generator().manual_seed(seed)
                            fallback_img = dig_helpers.run_pipe(
                                pipe=session.pipe,
                                prompt=[base_prompt],
                                generator_kwargs=gen_kwargs,
                                generator=generator,
                                num_images_per_prompt=1
                            )[0]
                        next_images.append(fallback_img)

                # Update session with new panel
                session.current_imgs = next_images
                session.current_seeds = next_seeds

                # Create new item keys and register items
                next_panel_no = session.iter + 1
                session.current_item_keys = []
                session.current_global_indices = []

                for i, (seed, img) in enumerate(zip(next_seeds, next_images)):
                    # Check if this (seed, prompt) has been seen before
                    seed_prompt_key = (seed, session.prompt)
                    if seed_prompt_key in session.seed_to_global:
                        global_idx = session.seed_to_global[seed_prompt_key]
                    else:
                        global_idx = session.global_item_counter
                        session.global_item_counter += 1
                        session.seed_to_global[seed_prompt_key] = global_idx

                    session.current_global_indices.append(global_idx)
                    key = f"panel{next_panel_no:02d}_slot{i}_seed{seed}_global{global_idx:03d}"
                    session.current_item_keys.append(key)
                    _register_item(session, key, img, base_prompt)
                    session.bt_scores[key] = 0.5

                step2_end_time = time.time()
                step2_duration = step2_end_time - step2_start_time
                step2_msg = f"   Step 2 (Candidate Sampling) completed: {N_CAND} candidates generated and evaluated in {step2_duration:.2f}s ({step2_duration/N_CAND:.2f}s per candidate)"
                logging.info(step2_msg)

                # Total time summary
                total_duration = step1_duration + step2_duration
                summary_msg = f"   Total interaction time: {total_duration:.2f}s (Step1: {step1_duration:.2f}s, Step2: {step2_duration:.2f}s)"
                logging.info(summary_msg)

            except Exception as e:
                error_msg = f"DesignLora evaluation failed: {str(e)}"
                logging.error(error_msg)

        # Store winner info
        try:
            session.prev_winner_seed = int(winner_seed)
            if hasattr(session, 'current_global_indices') and sel_idx < len(session.current_global_indices):
                session.prev_winner_global_idx = session.current_global_indices[sel_idx]
                winner_msg = f" Winner stored: seed={session.prev_winner_seed}, global_idx={session.prev_winner_global_idx}"
                logging.info(winner_msg)
            else:
                session.prev_winner_global_idx = None
        except Exception:
            session.prev_winner_seed = None
            session.prev_winner_global_idx = None

        # Update preference profile
        _update_preference_profile(session, session.prompt, winner_img)

        # Save session state
        _save_session_state(session)

        # Convert new images to base64
        image_data = []
        for img in session.current_imgs:
            image_data.append(image_to_base64(img))

        session.iter += 1
        logging.info(f"DesignLora interaction completed successfully")

        # Prepare response with timing information
        response_data = {
            "status": "success",
            "iteration": session.iter,
            "images": image_data,
            "seeds": session.current_seeds,
            "bt_scores": dict(list(session.bt_scores.items())[-4:]),
            "preference_profile": session.pref_profile
        }

        # Add timing information if available
        if 'step1_duration' in locals() and 'step2_duration' in locals():
            response_data["timing"] = {
                "step1_finetuning_sec": round(step1_duration, 2),
                "step2_sampling_sec": round(step2_duration, 2),
                "total_sec": round(step1_duration + step2_duration, 2)
            }

        return JSONResponse(response_data)

    except Exception as e:
        logging.error(f"DesignLora interaction failed: {str(e)}")
        api_logger = get_logger()
        api_logger.log_system_event(
            event_type="error",
            user_id=request.userID,
            data={"endpoint": "/interact/designlora", "request": request.dict()},
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finetune")
async def finetune_lora(request: FineTuneRequest):
    """
    Fine-tune a specific LoRA checkpoint with user selection and question.
    This endpoint includes the full interact_designlora logic:
    - Step 1: 20-candidate LoRA finetuning with VLM dual-question evaluation
    - Step 2: 8-candidate sampling with Expected Improvement (EI) scoring
    - Saves new checkpoint
    - Returns top 4 images (slot 0: winner regenerated, slots 1-3: top EI candidates)
    """
    try:
        api_logger = get_logger()

        logging.info(
            "finetune: user=%s checkpoint=%s version=%s selected=%s model=%s question=%r",
            request.userID, request.checkpoint_id, request.version,
            request.selectedIndex, request.imageModel, request.question,
        )

        # Get or create session
        session = get_or_create_session(request.userID, "designlora")

        # Auto-setup if needed (allows skipping /setup call)
        _auto_setup_if_needed(
            session=session,
            user_id=request.userID,
            prompt=request.prompt,
            image_model=request.imageModel,
            system="designlora"
        )

        # Update prompt if provided
        if request.prompt:
            session.prompt = request.prompt
            logging.info(f"Updated prompt: {request.prompt}")
        else:
            logging.info(f"No prompt provided in request, using session prompt: {session.prompt}")

        # Load or create checkpoint
        session.checkpoint_id = request.checkpoint_id
        checkpoint_dir = Path(CHECKPOINT_DIR) / request.userID / request.checkpoint_id
        checkpoint_metadata_file = checkpoint_dir / "metadata.json"

        if checkpoint_metadata_file.exists():
            # Load existing checkpoint (specific version or latest)
            checkpoint_metadata = _load_checkpoint(request.userID, request.checkpoint_id, session.pipe, version=request.version)
            loaded_version = checkpoint_metadata.get("loaded_version", {}).get("iteration", 0)
            
            # For branching: always use max iteration + 1 (don't overwrite)
            max_iteration = checkpoint_metadata.get("current_iteration", 0)
            current_iteration = max_iteration  # Start from max, not from loaded version
            
            if request.version:
                logging.info(f"Loaded checkpoint {request.checkpoint_id}, version: {request.version} (weights from iter_{loaded_version})")
                logging.info(f"Will save as iteration {current_iteration + 1} (branching from version {request.version})")
            else:
                logging.info(f"Loaded latest checkpoint {request.checkpoint_id}, iteration: {current_iteration}")
        else:
            # Create new checkpoint (will be saved after training)
            current_iteration = 0
            logging.info("Creating new checkpoint %s", request.checkpoint_id)

        logging.info("Fine-tuning checkpoint %s (iteration %s), prompt=%r",
                     request.checkpoint_id, current_iteration, session.prompt)

        # === OPTION A: Restore panel from item_keys (for image replacement feature) ===
        if request.current_item_keys:
            logging.info(f"Restoring panel from {len(request.current_item_keys)} item_keys...")
            restore_success = _restore_panel_from_keys(session, request.current_item_keys)

            if not restore_success:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to restore panel from item_keys. Some images may not exist in session."
                )

            logging.info(f"Panel restored successfully from item_keys")

        # Validate current images exist
        if not session.current_imgs:
            raise HTTPException(status_code=400, detail="No images to select from. Please call /generate first")

        sel_idx = request.selectedIndex
        if sel_idx < 0 or sel_idx >= len(session.current_imgs):
            raise HTTPException(status_code=400, detail="Invalid selection index")

        # Get train_weight from config (default: 5, where alpha = weight * rank)
        train_weight = session.config.get("opt_kwargs", {}).get("train_weight", 5)
        session.lora_scale = train_weight
        logging.info(f"Using train_weight={train_weight} for LoRA training")


        # Update selection
        session.selected_idx = sel_idx
        winner_img = session.current_imgs[sel_idx]
        winner_seed = session.current_seeds[sel_idx]
        winner_key = session.current_item_keys[sel_idx]

        # Build loser keys
        loser_keys = []
        for i in range(len(session.current_imgs)):
            if i != sel_idx:
                loser_keys.append(session.current_item_keys[i])

        # BT batch update with preference pairs
        if loser_keys:
            import re
            winner_global = None
            loser_globals = []

            winner_match = re.search(r'global(\d+)', winner_key)
            if winner_match:
                winner_global = f"global{winner_match.group(1)}"

            for lk in loser_keys:
                loser_match = re.search(r'global(\d+)', lk)
                if loser_match:
                    loser_globals.append(f"global{loser_match.group(1)}")

            if winner_global and loser_globals:
                preference_pairs = [(winner_global, lg) for lg in loser_globals]
                logging.info(f"BT Update: {len(preference_pairs)} preference pairs")
                try:
                    llm_surrogate.bt_update(preference_pairs)
                    logging.info(f"BT update successful")
                except Exception as e:
                    logging.error(f"BT update failed: {e}")

        # Recompute BT scores
        _recompute_bt_scores(session)

        # === DESIGNLORA: Full interact_designlora logic ===
        # NOTE: VLM uses ONLY current panel's 4 images as reference (not accumulated history)

        if session.optimizer and session.config:
            try:
                # Get recent evaluation data - use current panel only
                recent_eval_data = []
                if hasattr(session, 'current_item_keys') and session.current_item_keys:
                    for i, key in enumerate(session.current_item_keys):
                        if key in session.item_img and key in session.bt_scores and i < len(session.current_seeds):
                            recent_eval_data.append({
                                'key': key,
                                'image': session.item_img[key],
                                'score': session.bt_scores[key],
                                'seed': session.current_seeds[i]
                            })

                if not recent_eval_data:
                    raise HTTPException(status_code=400, detail="No reference data available")

                # Start 20-candidate evaluation and training
                step1_start_time = time.time()
                N_CANDIDATES_PER_CYCLE = 20
                total_training_steps = 0
                # Merge train_generator_kwargs with generator_kwargs (train overrides gen)
                gen_kwargs_base = session.config.get("generator_kwargs", {})
                train_kwargs = {**gen_kwargs_base, **session.config.get("train_generator_kwargs", {})}

                # Prepare reference data (shared by all candidates)
                ref_bt_scores = [ref_data['score'] for ref_data in recent_eval_data[:4]]

                # Build score labels
                score_labels = []
                letters = ['A', 'B', 'C', 'D']
                for j, ref_data in enumerate(recent_eval_data[:4]):
                    letter = letters[j]
                    score = ref_data['score'] * 100
                    score_labels.append(f"{letter}={score:.1f}%")
                score_info = " ".join(score_labels)

                # Create ranking
                scores = [recent_eval_data[i]['score'] for i in range(len(recent_eval_data[:4]))]
                sorted_indices = sorted(range(len(recent_eval_data[:4])), key=lambda i: recent_eval_data[i]['score'], reverse=True)

                ranking_parts = []
                i = 0
                while i < len(sorted_indices):
                    current_score = scores[sorted_indices[i]]
                    equal_indices = [sorted_indices[i]]
                    j = i + 1
                    while j < len(sorted_indices) and abs(scores[sorted_indices[j]] - current_score) < 0.001:
                        equal_indices.append(sorted_indices[j])
                        j += 1
                    if len(equal_indices) == 1:
                        ranking_parts.append(letters[equal_indices[0]])
                    else:
                        equal_letters = [letters[idx] for idx in equal_indices]
                        ranking_parts.append("= ".join(equal_letters))
                    i = j
                ranking = " > ".join(ranking_parts)

                # Generate pairwise comparisons
                pairwise_comps = []
                for i in range(len(sorted_indices)):
                    for j in range(i+1, len(sorted_indices)):
                        winner_idx = sorted_indices[i]
                        loser_idx = sorted_indices[j]
                        if abs(scores[winner_idx] - scores[loser_idx]) >= 0.001:
                            pairwise_comps.append(f"{letters[winner_idx]} over {letters[loser_idx]}")
                pairwise_str = ", ".join(pairwise_comps) if pairwise_comps else "No significant differences"

                # Create labeled reference images
                ref_images_labeled = []
                actual_winner_key = session.current_item_keys[session.selected_idx] if hasattr(session, 'selected_idx') else None

                for j, ref_data in enumerate(recent_eval_data[:4]):
                    try:
                        from image_utils import add_minimal_label_for_vlm
                        letter_label = letters[j]
                        is_winner = (ref_data['key'] == actual_winner_key)
                        img_for_vlm = add_minimal_label_for_vlm(ref_data['image'], letter_label, is_winner=is_winner)
                        ref_images_labeled.append(img_for_vlm)
                    except Exception as e:
                        ref_images_labeled.append(ref_data['image'])

                # Create VLM prompt
                combined_qtxt = (
                    "You will see five images in sequence. First four are reference images A, B, C, D. "
                    "The fifth image is the candidate for evaluation.\n"
                    f"Reference preference scores: {score_info} (higher=better).\n"
                    f"Ranking (bestworst): {ranking}.\n"
                    f"Pairwise preferences: {pairwise_str}.\n"
                    "Given the user's preference examples A–D (with WINNER/LOSER labels), would the user prefer the candidate (5th image)? Answer Yes or No.\n"
                    "Assume that the user is likely to prefer images similar to WINNER examples, and unlikely to prefer images similar to LOSER examples. "
                    "If the candidate is similar to both WINNER and OTHER examples, infer the preference based on subtle differences."
                    "would the user prefer the candidate (5th image)? Answer Yes or No."
                )

                winner_image = session.current_imgs[session.selected_idx] if hasattr(session, 'selected_idx') else None

                # Generate candidate seeds
                candidate_seeds = [random.randint(0, 99999) for _ in range(N_CANDIDATES_PER_CYCLE)]
                logging.info(f"DesignLora: Starting 20-candidate evaluation with question: '{request.question}'")

                # Train on 20 candidates
                for seed_idx, fixed_seed in enumerate(candidate_seeds):
                    try:
                        # Generate candidate image
                        gen_kwargs = session.config.get("generator_kwargs", {})
                        original_steps = gen_kwargs.get("num_inference_steps", 28)
                        clean_steps = max(1, original_steps // 2)

                        with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                            generator = torch.Generator().manual_seed(fixed_seed)
                            candidate_img = dig_helpers.run_pipe(
                                pipe=session.pipe,
                                prompt=[session.prompt],
                                generator_kwargs={
                                    "num_inference_steps": clean_steps,
                                    "guidance_scale": gen_kwargs.get("guidance_scale", 0.0),
                                    "height": gen_kwargs.get("height", 384),
                                    "width": gen_kwargs.get("width", 384),
                                },
                                generator=generator,
                                num_images_per_prompt=1
                            )[0]

                        # Create dual-question edit
                        qa_dual = [
                            {
                                "question": f"Does the candidate image better reflect the user's question '{request.question}' compared to the reference image? Answer Yes or No.",
                                "answer": "Yes",
                                "ref_images": [winner_image] if winner_image is not None else [],
                                "candidate_image": candidate_img,
                                "image_dim": (384, 384)
                            },
                            {
                                "question": combined_qtxt,
                                "answer": "Yes",
                                "ref_images": ref_images_labeled,
                                "candidate_image": candidate_img,
                                "image_dim": (384, 384)
                            }
                        ]

                        edit_dual = dig_pipeline.create_edit(
                            session.pipe, session.vlm, session.vlm_processor,
                            session.config, qa_dual, session.prompt
                        )

                        # Evaluate with VLM
                        loss_eval, meta_eval = dig_operators.loss_vlm_multiqa(
                            pipe=session.pipe,
                            edit=edit_dual,
                            generator_kwargs=train_kwargs,
                            pred_x0=candidate_img
                        )

                        question_prob = float(meta_eval["probs"][0].cpu().item())
                        combined_with_pref_prob = float(meta_eval["probs"][1].cpu().item())

                        question_weight = session.config.get("designlora_question_weight", 0.5)
                        pref_weight = session.config.get("designlora_pref_weight", 0.5)
                        final_combined_prob = (question_weight * question_prob + pref_weight * combined_with_pref_prob)

                        # Get noise
                        generator = torch.Generator(device=session.pipe.device).manual_seed(fixed_seed)
                        pipe_cls = dig_helpers.get_pipe_cls(session.pipe)
                        if pipe_cls == "flux":
                            latent_shape = dig_helpers.get_flux_latent_shape(session.pipe, train_kwargs, pack=True)
                        elif pipe_cls == "flux2":
                            # FLUX2 expects unpacked 4D latents as input (it packs internally)
                            latent_shape = dig_helpers.get_flux2_latent_shape(session.pipe, train_kwargs, pack=False)
                        elif pipe_cls == "qwen":
                            latent_shape = dig_helpers.get_qwen_latent_shape(session.pipe, train_kwargs, pack=True)
                        else:
                            latent_shape = (1, session.pipe.unet.config.in_channels,
                                          train_kwargs.get("height", 384)//8,
                                          train_kwargs.get("width", 384)//8)
                        init_noise = torch.randn(latent_shape, generator=generator,
                                                device=session.pipe.device, dtype=session.pipe.dtype)

                        # Train
                        torch.cuda.empty_cache()
                        base_loss = dig_pipeline.inner_loop(
                            session.pipe, edit_dual, train_kwargs, generator,
                            session.lora_scale, init_noise, total_training_steps
                        )

                        session.optimizer.zero_grad()
                        base_loss.backward()
                        session.optimizer.step()
                        session.global_step_counter += 1
                        total_training_steps += 1

                    except Exception as e:
                        logging.warning(f"Failed candidate {seed_idx+1}: {e}")
                        continue

                step1_end_time = time.time()
                step1_duration = step1_end_time - step1_start_time
                logging.info(f"Step 1 (LoRA Finetuning): {total_training_steps} steps in {step1_duration:.2f}s")

                # Save checkpoint after training
                new_iteration = current_iteration + 1
                
                # Track branching: record parent version
                parent_version = None
                if request.version:
                    parent_version = request.version  # Branching from specific version
                elif checkpoint_metadata_file.exists():
                    parent_version = checkpoint_metadata.get("current_iteration", 0)  # Continuing from latest
                
                checkpoint_path = _save_checkpoint(
                    user_id=request.userID,
                    checkpoint_id=request.checkpoint_id,
                    pipe=session.pipe,
                    iteration=new_iteration,
                    question=request.question,
                    metadata={
                        "global_step": session.global_step_counter,
                        "training_steps": total_training_steps,
                        "parent_version": parent_version  # Track branching
                    }
                )
                logging.info(f"Checkpoint saved: {checkpoint_path}")

                # Save reference images for debugging
                try:
                    checkpoint_dir = Path(CHECKPOINT_DIR) / request.userID / request.checkpoint_id
                    ref_images_dir = checkpoint_dir / f"reference_images_iter_{new_iteration:03d}"
                    ref_images_dir.mkdir(parents=True, exist_ok=True)

                    # Save winner image
                    if winner_image is not None:
                        winner_path = ref_images_dir / "winner.png"
                        winner_image.save(winner_path)
                        logging.info(f"Saved winner image: {winner_path}")

                    # Save labeled reference images (A, B, C, D)
                    for idx, ref_img in enumerate(ref_images_labeled):
                        ref_path = ref_images_dir / f"ref_{chr(65+idx)}.png"  # A, B, C, D
                        ref_img.save(ref_path)

                    if ref_images_labeled:
                        logging.info(f"Saved {len(ref_images_labeled)} labeled reference images")

                    # Save metadata
                    ref_metadata = {
                        "iteration": new_iteration,
                        "question": request.question,
                        "selected_index": session.selected_idx if hasattr(session, 'selected_idx') else None,
                        "num_ref_images": len(ref_images_labeled),
                        "timestamp": datetime.now().isoformat()
                    }
                    ref_metadata_path = ref_images_dir / "metadata.json"
                    with open(ref_metadata_path, 'w') as f:
                        json.dump(ref_metadata, f, indent=2)

                except Exception as e:
                    logging.warning(f"Failed to save reference images: {e}")

                # === Step 2: Generate new candidates with EI scoring ===
                step2_start_time = time.time()
                N_CAND = 8
                N_SHOW = 4

                torch.cuda.empty_cache()

                cand_seeds = [random.randint(0, 99999) for _ in range(N_CAND)]
                stats = []

                logging.info(f"Generating {N_CAND} new candidates with trained LoRA...")

                for s in cand_seeds:
                    try:
                        gen_kwargs = session.config.get("generator_kwargs", {})
                        with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                            generator = torch.Generator().manual_seed(s)
                            cand_img = dig_helpers.run_pipe(
                                pipe=session.pipe,
                                prompt=[session.prompt],
                                generator_kwargs=gen_kwargs,
                                generator=generator,
                                num_images_per_prompt=1
                            )[0]

                        # Evaluate candidate
                        qa_dual_eval = [
                            {
                                "question": f"Does the candidate image better reflect the user's question '{request.question}' compared to the reference image? Answer Yes or No.",
                                "answer": "Yes",
                                "ref_images": [winner_image] if winner_image is not None else [],
                                "candidate_image": cand_img,
                                "image_dim": (384, 384)
                            },
                            {
                                "question": combined_qtxt,
                                "answer": "Yes",
                                "ref_images": ref_images_labeled,
                                "candidate_image": cand_img,
                                "image_dim": (384, 384)
                            }
                        ]

                        edit_dual_eval = dig_pipeline.create_edit(
                            session.pipe, session.vlm, session.vlm_processor,
                            session.config, qa_dual_eval, session.prompt
                        )

                        # Multiple evaluations
                        N_EVAL = 3
                        yes_probs = []
                        for eval_idx in range(N_EVAL):
                            try:
                                loss_eval, meta_eval = dig_operators.loss_vlm_multiqa(
                                    pipe=session.pipe,
                                    edit=edit_dual_eval,
                                    generator_kwargs=gen_kwargs,
                                    pred_x0=cand_img
                                )
                                question_prob = float(meta_eval["probs"][0].cpu().item())
                                combined_with_pref_prob = float(meta_eval["probs"][1].cpu().item())
                                eval_prob_yes = (question_weight * question_prob + pref_weight * combined_with_pref_prob)
                                yes_probs.append(eval_prob_yes)
                            except Exception as e:
                                yes_probs.append(0.5)

                        if yes_probs:
                            mu = np.mean(yes_probs)
                            sigma = np.std(yes_probs) + 0.01
                            prob_yes = mu
                        else:
                            mu, sigma = 0.5, 0.1
                            prob_yes = 0.5

                        import math
                        if 0.001 < prob_yes < 0.999:
                            yes_logit = math.log(prob_yes / (1 - prob_yes))
                        else:
                            yes_logit = 0.0
                        stats.append((s, mu, sigma, cand_img, prob_yes, yes_logit))

                    except Exception as e:
                        stats.append((s, 0.5, 0.1, None, 0.5, 0.0))

                # EI scoring
                def expected_improvement(mu, sigma, best=0.5, xi=0.0):
                    from scipy.stats import norm
                    if sigma == 0:
                        return 0.0
                    z = (mu - best - xi) / sigma
                    ei = (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                    return max(0.0, ei)

                scored = []
                for stat in stats:
                    seed, mu, sigma, img, token_prob, logit = stat
                    ei_score = expected_improvement(mu, sigma, best=0.5, xi=0.0)
                    scored.append((seed, ei_score, mu, sigma, img, token_prob, logit))

                sorted_scored = sorted(scored, key=lambda x: -x[1])

                # Build next panel
                next_seeds = []
                next_images = []

                # Slot 0: Previous winner with updated LoRA
                next_seeds.append(winner_seed)
                with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                    generator = torch.Generator().manual_seed(winner_seed)
                    winner_img_new = dig_helpers.run_pipe(
                        pipe=session.pipe,
                        prompt=[session.prompt],
                        generator_kwargs=gen_kwargs,
                        generator=generator,
                        num_images_per_prompt=1
                    )[0]
                next_images.append(winner_img_new)

                # Slots 1-3: Top EI candidates
                for item in sorted_scored[:N_SHOW-1]:
                    seed, ei_score, mu, sigma, img, token_prob, logit = item
                    next_seeds.append(seed)
                    if img is not None:
                        next_images.append(img)
                    else:
                        with dig_helpers.LoraManager(session.pipe, enter_weights=session.lora_scale):
                            generator = torch.Generator().manual_seed(seed)
                            fallback_img = dig_helpers.run_pipe(
                                pipe=session.pipe,
                                prompt=[session.prompt],
                                generator_kwargs=gen_kwargs,
                                generator=generator,
                                num_images_per_prompt=1
                            )[0]
                        next_images.append(fallback_img)

                # Update session
                session.current_imgs = next_images
                session.current_seeds = next_seeds

                # Register new items
                next_panel_no = session.iter + 1
                session.current_item_keys = []
                session.current_global_indices = []

                for i, (seed, img) in enumerate(zip(next_seeds, next_images)):
                    # Use (seed, prompt) as key
                    seed_prompt_key = (seed, session.prompt)
                    if seed_prompt_key in session.seed_to_global:
                        global_idx = session.seed_to_global[seed_prompt_key]
                    else:
                        global_idx = session.global_item_counter
                        session.global_item_counter += 1
                        session.seed_to_global[seed_prompt_key] = global_idx

                    session.current_global_indices.append(global_idx)
                    key = f"panel{next_panel_no:02d}_slot{i}_seed{seed}_global{global_idx:03d}"
                    session.current_item_keys.append(key)
                    _register_item(session, key, img, session.prompt)
                    session.bt_scores[key] = 0.5

                step2_end_time = time.time()
                step2_duration = step2_end_time - step2_start_time
                logging.info(f"Step 2 (Candidate Sampling): {N_CAND} candidates in {step2_duration:.2f}s")

            except Exception as e:
                logging.error(f"DesignLora fine-tuning failed: {str(e)}")
                raise

        # Update preference profile
        _update_preference_profile(session, session.prompt, winner_img)

        # Store winner info
        session.prev_winner_seed = int(winner_seed)
        if hasattr(session, 'current_global_indices') and sel_idx < len(session.current_global_indices):
            session.prev_winner_global_idx = session.current_global_indices[sel_idx]

        # Save session state
        _save_session_state(session)

        # Convert images to base64
        image_data = [image_to_base64(img) for img in session.current_imgs]

        session.iter += 1

        # Prepare response
        response_data = {
            "status": "success",
            "checkpoint_id": request.checkpoint_id,
            "iteration": new_iteration,
            "images": image_data,
            "seeds": session.current_seeds,
            "item_keys": session.current_item_keys,  # NEW: 각 이미지의 고유 식별자
            "bt_scores": dict(list(session.bt_scores.items())[-4:]),
            "preference_profile": session.pref_profile
        }

        if 'step1_duration' in locals() and 'step2_duration' in locals():
            response_data["timing"] = {
                "step1_finetuning_sec": round(step1_duration, 2),
                "step2_sampling_sec": round(step2_duration, 2),
                "total_sec": round(step1_duration + step2_duration, 2)
            }

        return JSONResponse(response_data)

    except FileNotFoundError as e:
        logging.error(f"Checkpoint not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Fine-tuning failed: {str(e)}")
        api_logger = get_logger()
        api_logger.log_system_event(
            event_type="error",
            user_id=request.userID,
            data={"endpoint": "/finetune", "request": request.dict()},
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_with_lora")
async def generate_with_lora(request: GenerateWithLoRARequest):
    """Generate images using a specific LoRA checkpoint (no training)."""
    try:
        api_logger = get_logger()

        logging.info(
            "generate_with_lora: user=%s checkpoint=%s version=%s prompt=%r n=%s model=%s",
            request.userID, request.checkpoint_id, request.version,
            request.prompt, request.num_images, request.imageModel,
        )

        # Get or create session
        session = get_or_create_session(request.userID, "designlora")

        # Auto-setup if needed (allows skipping /setup call)
        _auto_setup_if_needed(
            session=session,
            user_id=request.userID,
            prompt=request.prompt,
            image_model=request.imageModel,
            system="designlora"
        )

        # Load checkpoint if exists, otherwise use base LoRA
        session.checkpoint_id = request.checkpoint_id
        checkpoint_dir = Path(CHECKPOINT_DIR) / request.userID / request.checkpoint_id
        checkpoint_metadata_file = checkpoint_dir / "metadata.json"

        checkpoint_info_msg = ""
        if checkpoint_metadata_file.exists():
            # Debug: Show available versions before loading
            with open(checkpoint_metadata_file, 'r') as f:
                raw_metadata = json.load(f)
            available_versions = [v['iteration'] for v in raw_metadata.get('versions', [])]
            logging.info(f"Available versions in metadata: {available_versions}")

            # Load existing checkpoint (specific version or latest)
            checkpoint_metadata = _load_checkpoint(request.userID, request.checkpoint_id, session.pipe, version=request.version)
            session.lora_scale = 1.0  # Full LoRA strength for generation

            iteration = checkpoint_metadata.get('iteration', 0)
            which = f"version {request.version}" if request.version is not None else "latest"
            checkpoint_info_msg = f"Using checkpoint '{request.checkpoint_id}' ({which}, iteration {iteration})"
            logging.info(checkpoint_info_msg)
        else:
            # Checkpoint doesn't exist - fall back to the base model (no LoRA yet).
            checkpoint_metadata = {"iteration": 0}
            session.lora_scale = 0.0
            checkpoint_info_msg = f"Checkpoint '{request.checkpoint_id}' not found - using base model (no LoRA)"
            logging.info(checkpoint_info_msg)

        logging.info("Generating %s images...", request.num_images)

        # Generate images with the loaded LoRA
        gen_kwargs = session.config.get("generator_kwargs", {})

        # Generate 4 seeds based on base_seed
        if request.num_images == 4:
            if request.seed is not None:
                # Client provided base seed: create variants
                base_seed = request.seed
                fixed_seeds = [base_seed, base_seed + 10, base_seed + 20, base_seed + 30]
                logging.info(f"Using base seed {base_seed}  seeds: {fixed_seeds}")
            else:
                # No seed provided: use default
                fixed_seeds = [10, 20, 30, 40]
                logging.info(f"Using default seeds: {fixed_seeds}")

            images, seeds = generate_images_simple(
                pipe=session.pipe,
                prompt=request.prompt,
                num_images=request.num_images,
                seed=None,
                lora_scale=session.lora_scale,
                gen_kwargs=gen_kwargs,
                seed_list=fixed_seeds
            )
        else:
            # For non-4 images, use provided seed or random
            images, seeds = generate_images_simple(
                pipe=session.pipe,
                prompt=request.prompt,
                num_images=request.num_images,
                seed=request.seed,
                lora_scale=session.lora_scale,
                gen_kwargs=gen_kwargs
            )

        logging.info(f"Generated {len(images)} images with seeds: {seeds}")

        # ===== REGISTER IMAGES IN SESSION (for History feature) =====
        session.current_imgs = images
        session.current_seeds = seeds
        session.current_item_keys = []
        session.current_global_indices = []

        for i, (seed, img) in enumerate(zip(seeds, images)):
            # Create global index for new (seed, prompt) combinations
            seed_prompt_key = (seed, request.prompt)
            if seed_prompt_key in session.seed_to_global:
                global_idx = session.seed_to_global[seed_prompt_key]
            else:
                global_idx = session.global_item_counter
                session.global_item_counter += 1
                session.seed_to_global[seed_prompt_key] = global_idx

            session.current_global_indices.append(global_idx)
            # Use a special panel marker for generate_with_lora (not tied to iteration)
            key = f"gen_lora_slot{i}_seed{seed}_global{global_idx:03d}"
            session.current_item_keys.append(key)
            _register_item(session, key, img, request.prompt)
            session.bt_scores[key] = 0.5

        logging.info(f"Registered {len(session.current_item_keys)} images with item_keys: {session.current_item_keys}")

        # ===== SAVE IMAGES TO CHECKPOINT DIRECTORY =====
        try:
            # Save to checkpoint directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_version = checkpoint_metadata.get('iteration', 0)

            checkpoint_dir = Path(CHECKPOINT_DIR) / request.userID / request.checkpoint_id
            generated_dir = checkpoint_dir / f"generated_images_v{checkpoint_version}_{timestamp}"
            generated_dir.mkdir(parents=True, exist_ok=True)

            # Save images with seed in filename
            saved_paths = []
            for seed, img in zip(seeds, images):
                img_path = generated_dir / f"image_seed_{seed:05d}.png"
                img.save(img_path)
                saved_paths.append(str(img_path))

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint_id": request.checkpoint_id,
                "checkpoint_version": checkpoint_version,
                "prompt": request.prompt,
                "num_images": request.num_images,
                "seeds": seeds,
                "item_keys": session.current_item_keys,
                "lora_scale": session.lora_scale,
                "generation_type": "generate_with_lora"
            }
            metadata_path = generated_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logging.info("Saved %s images to %s (checkpoint %s v%s)",
                         len(saved_paths), generated_dir, request.checkpoint_id, checkpoint_version)
        except Exception as e:
            logging.warning("Failed to save images to disk: %s", e)

        # Convert images to base64
        image_data = [image_to_base64(img) for img in images]

        # Log generation
        api_logger.log_system_event(
            event_type="generate_with_lora",
            user_id=request.userID,
            data={
                "checkpoint_id": request.checkpoint_id,
                "checkpoint_iteration": checkpoint_metadata.get("iteration", 0),
                "version": request.version,
                "prompt": request.prompt,
                "num_images": request.num_images,
                "seeds": seeds,
                "lora_scale": session.lora_scale,
                "checkpoint_path": checkpoint_metadata.get('checkpoint_path', 'none'),
                "item_keys": session.current_item_keys
            }
        )

        logging.info("generate_with_lora completed: %s items", len(session.current_item_keys))

        return JSONResponse({
            "status": "success",
            "checkpoint_id": request.checkpoint_id,
            "checkpoint_iteration": checkpoint_metadata.get("iteration", 0),
            "images": image_data,
            "seeds": seeds,
            "item_keys": session.current_item_keys,  # For History feature
            "prompt": request.prompt
        })

    except FileNotFoundError as e:
        logging.error(f"Checkpoint not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Generation with LoRA failed: {str(e)}")
        api_logger = get_logger()
        api_logger.log_system_event(
            event_type="error",
            user_id=request.userID,
            data={"endpoint": "/generate_with_lora", "request": request.dict()},
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/checkpoints/{userID}")
async def get_checkpoints(userID: str):
    """Get all LoRA checkpoints for a user."""
    try:
        checkpoints = _list_checkpoints(userID)

        return JSONResponse({
            "status": "success",
            "userID": userID,
            "checkpoints": [ckpt.dict() for ckpt in checkpoints],
            "total": len(checkpoints)
        })

    except Exception as e:
        logging.error(f"Failed to list checkpoints for user {userID}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "DesignLoRA Server"}

# ============= Main =============
if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
