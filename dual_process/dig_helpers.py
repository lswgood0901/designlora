# -----------------------------------------------------------------------------
# Adapted from Dual-Process Image Generation (g-luo/dual_process, ICCV 2025).
# DesignLoRA additions/changes are tagged "[DesignLoRA]"; see
# dual_process/UPSTREAM_CHANGES.md for the full list.
# -----------------------------------------------------------------------------
import diffusers
import numpy as np
from omegaconf import OmegaConf
import torch
import transformers
import accelerate

from diffusers.training_utils import compute_density_for_timestep_sampling
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoProcessor


# ===========================
#        Load Models
# ===========================
def check_vram(device, min_vram_gb=None):
    if min_vram_gb is not None:
        free, total = torch.cuda.mem_get_info(device)
        if free / 1024**3 < min_vram_gb:
            raise ValueError(f"Not enough VRAM on device {device}. Please set pipe_kwargs.pipe_device and vlm_kwargs.vlm_device to different GPUs. You need at least {min_vram_gb}GB of VRAM.")

def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def load_vlm(vlm_device, vlm_id, vlm_cls, **vlm_kwargs):
    check_vram(vlm_device, vlm_kwargs.pop("min_vram_gb", None))
    vlm_kwargs = {k: eval(v) if ("dtype" in k and v != "auto") else v for k, v in vlm_kwargs.items()}
    
    # Handle load_in_4bit: if False, remove it to avoid passing it to models that don't support it in __init__
    if vlm_kwargs.get("load_in_4bit") is False:
        vlm_kwargs.pop("load_in_4bit")
        
    vlm_cls = getattr(transformers, vlm_cls)
    vlm = vlm_cls.from_pretrained(vlm_id, device_map=vlm_device, **vlm_kwargs)
    vlm_processor = AutoProcessor.from_pretrained(vlm_id)
    # Disable image splitting
    if hasattr(vlm_processor.image_processor, "do_image_splitting"):
        vlm_processor.image_processor.do_image_splitting = False
    # Turn off requires grad
    set_requires_grad(vlm, False)
    return vlm, vlm_processor

def load_pipe(pipe_devices, pipe_id, pipe_cls, scheduler_config={}, **pipe_kwargs):
    """
    Load a diffusion pipeline across multiple GPUs using Accelerate's
    automatic layer balancing.

    Parameters
    ----------
    pipe_devices : list[int]
        GPU indices, e.g. [0, 1, 2, 3].
    pipe_id : str
        Local path or HF repo of the checkpoint.
    pipe_cls : str
        Name of the Pipeline class inside diffusers, e.g. ``"FluxPipeline"``.
    scheduler_config : dict, optional
        Same semantics as before.
    **pipe_kwargs
        Extra keyword‑args forwarded to ``from_pretrained``.
        Special key ``gpu_ram_limit`` (GiB) controls per‑GPU memory cap.
    """
    # ---- VRAM check (optional) ------------------------------------------------
    min_vram_gb = pipe_kwargs.pop("min_vram_gb", None)
    if min_vram_gb is not None:
        for d in pipe_devices:
            check_vram(f"cuda:{d}", min_vram_gb)

    # ---- Parse dtype strings --------------------------------------------------
    pipe_kwargs = {k: eval(v) if ("dtype" in k and v != "auto") else v
                   for k, v in pipe_kwargs.items()}

    # ---- Build device_map / max_memory ---------------------------------------
    gpu_ram_limit = pipe_kwargs.pop("gpu_ram_limit", "92")  # GiB (str or number)
    max_memory = {d: f"{gpu_ram_limit}GiB" for d in pipe_devices}

    # ---- Instantiate pipeline -------------------------------------------------
    pipe_cls_obj = getattr(diffusers, pipe_cls)

    # Special handling for Qwen: ensure text_encoder is loaded properly
    is_qwen_pipeline = False
    if "Qwen" in pipe_cls or pipe_cls == "DiffusionPipeline":
        # Check if this is actually a Qwen model by looking at model_index.json
        import json
        from pathlib import Path
        model_index_path = Path(pipe_id) / "model_index.json"
        if model_index_path.exists():
            with open(model_index_path) as f:
                model_index = json.load(f)
            if model_index.get("_class_name") == "QwenImagePipeline":
                is_qwen_pipeline = True

    # For Qwen-Image, load without device_map to avoid Accelerate hooks
    if is_qwen_pipeline:
        target_device = f"cuda:{pipe_devices[0]}"

        qwen_kwargs = {
            "torch_dtype": pipe_kwargs.get("torch_dtype", torch.bfloat16),
            "local_files_only": True,
        }

        pipe = pipe_cls_obj.from_pretrained(pipe_id, **qwen_kwargs)
        pipe = pipe.to(target_device)

        # For Qwen: Don't modify scheduler, don't reload tokenizer
        # Just set progress bar and freeze parameters
        pipe.set_progress_bar_config(disable=True)

        # Freeze parameters
        for module_name in pipe.config.keys():
            module = getattr(pipe, module_name)
            if isinstance(module, torch.nn.Module):
                set_requires_grad(module, False)

        pipe.scheduler_config = scheduler_config
        return pipe

    else:
        # For other pipelines, use device_map from config or default to "balanced"
        device_map = pipe_kwargs.pop("device_map", "balanced")
        # Use max_memory from config if provided, otherwise use generated one
        config_max_memory = pipe_kwargs.pop("max_memory", None)
        if config_max_memory is not None:
            # Convert string keys to int (YAML keys come as strings)
            max_memory = {int(k): v for k, v in config_max_memory.items()}
        pipe = pipe_cls_obj.from_pretrained(
            pipe_id,
            device_map=device_map,
            max_memory=max_memory,
            **pipe_kwargs
        )

    # ---- Post‑processing (for non-Qwen pipelines) ------------------------------
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.safety_checker = None

    # Freeze parameters
    for module_name in pipe.config.keys():
        module = getattr(pipe, module_name)
        if isinstance(module, torch.nn.Module):
            set_requires_grad(module, False)

    # Replace scheduler if requested (for non-Qwen pipelines only)
    scheduler_cls = (scheduler_config.get("eval_scheduler_cls")
                     or scheduler_config.get("train_scheduler_cls"))
    if scheduler_cls is not None:
        pipe.scheduler = getattr(diffusers, scheduler_cls).from_config(
            pipe.scheduler.config
        )

    pipe.scheduler_config = scheduler_config
    return pipe

def load_config(config_names):
    if type(config_names) is not list:
        config_names = [config_names]
    config = OmegaConf.create({})
    for config_name in config_names:
        config = OmegaConf.merge(config, OmegaConf.load(config_name))
    config = OmegaConf.to_container(config, resolve=True)
    return config

# ===========================
#        LoRA Helpers
# ===========================
def get_pipe_cls(pipe):
    class_name = str(type(pipe))
    if "Flux2" in class_name:
        return "flux2"
    elif "Flux" in class_name:
        return "flux"
    elif "StableDiffusion" in class_name:
        return "sd"
    elif "Sana" in class_name:
        return "sana"
    elif "Qwen" in class_name or "DiffusionPipeline" in class_name: # Fallback for generic pipeline if Qwen
        return "qwen"
    else:
        return None
    
def get_backbone(pipe):
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    elif hasattr(pipe, "unet"):
        return pipe.unet
    else:
        return pipe
    
def create_lora(pipe, lora_lr, lora_name="default", **lora_kwargs):
    backbone = get_backbone(pipe)
    # Only delete and recreate if the adapter doesn't exist or if explicitly requested
    if not hasattr(backbone, 'peft_config') or lora_name not in backbone.peft_config:
        # Adapter doesn't exist, create it
        backbone.delete_adapters(lora_name)  # Safe to call even if doesn't exist
        backbone.add_adapter(
            LoraConfig(**lora_kwargs), 
            adapter_name=lora_name
        )
    # Return the existing or newly created LoRA parameters
    params = {"params": [p for name, p in backbone.named_parameters() if "lora" in name], "lr": lora_lr}
    return params

def toggle_lora(pipe, weights):
    backbone = get_backbone(pipe)
    if hasattr(backbone, "peft_config"):
        adapter_names = backbone.active_adapters()
        backbone.set_adapters(adapter_names=adapter_names, weights=weights)

class LoraManager:
    def __init__(self, pipe, enter_weights=1, exit_weights=1):
        self.pipe = pipe
        self.enter_weights = enter_weights
        self.exit_weights = exit_weights

    def __enter__(self):
        toggle_lora(self.pipe, self.enter_weights)

    def __exit__(self, exc_type, exc_value, traceback):
        toggle_lora(self.pipe, self.exit_weights)

def save_weights(pipe, output_file):
    backbone = get_backbone(pipe)
    if hasattr(backbone, "peft_config"):
        torch.save(get_peft_model_state_dict(backbone), f"{output_file}.pt")

def load_weights(pipe, output_file):
    backbone = get_backbone(pipe)
    state_dict = torch.load(f"{output_file}.pt")
    state_dict = {n.replace(".weight", ".default.weight"): p for n, p in state_dict.items()}
    backbone.load_state_dict(state_dict, strict=False)

# ===========================
#        Edit Helpers
# ===========================
def create_edit(pipe, vlm, vlm_processor, cfg=None):
    """
    Construct a lightweight 'edit' dictionary shared by app_bt.py ▸ dig_pipeline.py.

    Parameters
    ----------
    pipe : diffusers.Pipeline
        The diffusion pipeline currently in use (with or without LoRA weights).
    vlm : transformers.PreTrainedModel
        Vision‑Language Model used for VQA / distillation losses.
    vlm_processor : transformers.AutoProcessor
        Paired processor for the VLM (handles image+text tokenisation).
    cfg : dict | OmegaConf | None
        Optional experiment‑wide config for reference (kept as‑is).

    Returns
    -------
    dict
        A simple container with standardised keys so that downstream calls
        (e.g. `loss_vlm`, `loss_bt_vlm`) receive everything they need without
        tight coupling to the outer scope.
    """
    return {
        "pipe": pipe,
        "vlm": vlm,
        "vlm_processor": vlm_processor,
        "cfg": cfg,
    }

# ===========================
#  General Pipeline Helpers
# ===========================
def renormalize(x, range_a, range_b):
    min_a, max_a = range_a
    min_b, max_b = range_b
    return ((x - min_a) / (max_a - min_a)) * (max_b - min_b) + min_b

def create_callback_interrupt(stop_i):
    # Runs stop_i steps then early stops
    def callback_on_step_end(self, i, t, callback_kwargs):
        assert stop_i > 0
        if i >= (stop_i - 1):
            self._interrupt = True
        else:
            self._interrupt = False
        return {}
    return callback_on_step_end

@torch.no_grad()
def run_pipe(pipe, generator_kwargs, prompt_kwargs={}, stop_i=None, **kwargs):
    pipe_kwargs = {**generator_kwargs, **kwargs}
    pipe_cls = get_pipe_cls(pipe)

    # Special handling for Qwen with separate positive/negative embeddings
    if pipe_cls == "qwen" and "negative_encoder_hidden_states" in prompt_kwargs:
        # Qwen stores positive and negative separately (from encode_qwen_text)
        pipe_kwargs["prompt_embeds"] = prompt_kwargs["encoder_hidden_states"]
        pipe_kwargs["prompt_embeds_mask"] = prompt_kwargs["encoder_hidden_states_mask"]
        pipe_kwargs["negative_prompt_embeds"] = prompt_kwargs["negative_encoder_hidden_states"]
        pipe_kwargs["negative_prompt_embeds_mask"] = prompt_kwargs["negative_encoder_hidden_states_mask"]
        pipe_kwargs["prompt"] = None
        pipe_kwargs["negative_prompt"] = None
    else:
        # Standard handling for other pipelines
        prompt_remapping = {
            "pooled_projections": "pooled_prompt_embeds",
            "encoder_hidden_states": "prompt_embeds",
            "encoder_attention_mask": "prompt_attention_mask",
            "encoder_hidden_states_mask": "prompt_attention_mask",  # For Qwen
        }
        for k, v in prompt_remapping.items():
            if k in prompt_kwargs:
                pipe_kwargs[v] = prompt_kwargs[k]

        if "prompt_embeds" in pipe_kwargs:
            prompt_embeds = pipe_kwargs["prompt_embeds"]
            # Handle classifier-free guidance
            assert prompt_embeds.shape[0] <= 2, "Only one prompt supported in this mode due to cfg handling"
            if prompt_embeds.shape[0] == 2:
                pipe_kwargs["prompt_embeds"] = prompt_embeds[1][None, ...]
                pipe_kwargs["negative_prompt_embeds"] = prompt_embeds[0][None, ...]
                if "prompt_attention_mask" in pipe_kwargs:
                    prompt_attention_mask = pipe_kwargs["prompt_attention_mask"]
                    pipe_kwargs["prompt_attention_mask"] = prompt_attention_mask[1][None, ...]
                    pipe_kwargs["negative_prompt_attention_mask"] = prompt_attention_mask[0][None, ...]
                pipe_kwargs["prompt"] = None
                pipe_kwargs["negative_prompt"] = None

    # Special handling for Qwen pipeline (match generate_image.py exactly)
    if pipe_cls == "qwen":
        # Qwen expects string prompts, not lists
        if "prompt" in pipe_kwargs and isinstance(pipe_kwargs["prompt"], list):
            pipe_kwargs["prompt"] = pipe_kwargs["prompt"][0]

        # Qwen-Image requires negative_prompt (even if empty string) when not using embeddings
        if "negative_prompt" not in pipe_kwargs and "negative_prompt_embeds" not in pipe_kwargs:
            pipe_kwargs["negative_prompt"] = " "  # Empty string as per official example

        # Qwen uses true_cfg_scale, NOT guidance_scale
        # Remove guidance_scale and use true_cfg_scale only (like generate_image.py)
        true_cfg = pipe_kwargs.pop("guidance_scale", 4.0)
        if "true_cfg_scale" not in pipe_kwargs:
            pipe_kwargs["true_cfg_scale"] = true_cfg

    if stop_i is not None:
        callback_interrupt = create_callback_interrupt(stop_i)
        pipe_kwargs["callback_on_step_end"] = callback_interrupt

    if stop_i is None or stop_i > 0:
        return pipe(**pipe_kwargs).images
    else:
        # stop_i == 0: Return initial latents without running pipeline
        latents = pipe_kwargs.get("latents")
        # For FLUX2, we need to pack the latents since the pipeline would normally do this
        # but we're bypassing it. The transformer expects packed 3D: (B, H*W, C)
        if pipe_cls == "flux2" and latents is not None and latents.ndim == 4:
            latents = pipe._pack_latents(latents)
        return latents

def get_train_scheduler(pipe):
    scheduler_cls = pipe.scheduler_config["train_scheduler_cls"]
    scheduler_obj = getattr(diffusers, scheduler_cls)

    # Create a dummy scheduler to get valid config keys from pipe.scheduler.config
    scheduler_config = pipe.scheduler.config
    dummy_scheduler = scheduler_obj()
    valid_keys = set(dummy_scheduler.config.keys())
    valid_config = {k: v for k, v in scheduler_config.items() if k in valid_keys}
    
    # Create scheduler
    scheduler = scheduler_obj(**valid_config)
    scheduler._class_name = scheduler_cls

    # Set timesteps
    num_timesteps = pipe.scheduler_config["num_train_timesteps"]
    if scheduler_cls == "FlowMatchEulerDiscreteScheduler":
        sigmas = np.linspace(1.0, 1 / num_timesteps, num_timesteps)

        # If dynamic shifting is enabled, FlowMatchEulerDiscreteScheduler
        # requires a `mu` value.  Provide a default (0.0) when none exists.
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            mu = getattr(scheduler.config, "mu", 0.0)
            scheduler.set_timesteps(sigmas=sigmas, mu=mu)
        else:
            scheduler.set_timesteps(sigmas=sigmas)
    else:
        scheduler.set_timesteps(num_timesteps)
    return scheduler

def get_timestep(pipe):
    train_scheduler = get_train_scheduler(pipe)
    timesteps = train_scheduler.timesteps.to(pipe.device)
    # (1) 원하는 sigmas 정의
    # num_steps = train_scheduler.config.num_train_timesteps
    # sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    # # (2) 스케줄러에 적용
    # if getattr(train_scheduler.config, "use_dynamic_shifting", False):
    #     mu = getattr(train_scheduler.config, "mu", 0.0)   # 없으면 0.0
    #     train_scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    # else:
    #     train_scheduler.set_timesteps(sigmas=sigmas)

    # # (3) **적용 후** timesteps 읽기
    # timesteps = train_scheduler.timesteps.to(pipe.device)

    # (4) 샘플링
    u = compute_density_for_timestep_sampling(
        weighting_scheme="none",
        batch_size=1,
    )
    indices = (u * len(timesteps)).long()
    t = timesteps[indices]
    i = indices.long().item()
    return i, t

# ===========================
#    Flux Pipeline Helpers
# ===========================
def get_flux_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    backbone = get_backbone(pipe)
    num_channels_latents = backbone.config.in_channels // 4
    height, width = generator_kwargs["height"], generator_kwargs["width"]
    height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    if not pack:
        shape = (batch_size, num_channels_latents, height, width)
    else:
        shape = (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return shape

def encode_flux_text(pipe, prompts, keys, device):
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device
        )
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
        }
    return prompt_kwargs

# ===========================
#   Flux2 Pipeline Helpers
# ===========================
# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def encode_flux2_text(pipe, prompts, keys, device):
    """
    Encode text for Flux2 pipeline.

    Flux2 uses Mistral3 as text encoder and returns:
    - prompt_embeds: (batch, seq_len, hidden_size)
    - text_ids: position IDs for text tokens

    No pooled_prompt_embeds unlike FLUX.1!
    """
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=device
        )
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
        }
    return prompt_kwargs

# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def get_flux2_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    """Get latent shape for Flux2 pipeline.

    FLUX2's prepare_latents creates:
    - Unpacked 4D: (B, num_channels*4, h//2, w//2) = (1, 128, 24, 24) for 384x384 input
    - Packed 3D: (B, (h//2)*(w//2), num_channels*4) = (1, 576, 128)

    Where:
    - num_channels = transformer.config.in_channels // 4 = 128 // 4 = 32
    - h, w = 2 * (image_size // (vae_scale_factor * 2)) = 2 * (384 // 16) = 48
    """
    backbone = get_backbone(pipe)
    num_channels_latents = backbone.config.in_channels // 4  # 128 // 4 = 32
    height, width = generator_kwargs["height"], generator_kwargs["width"]
    # Calculate intermediate height/width (same as FLUX2 pipeline)
    h = 2 * (int(height) // (pipe.vae_scale_factor * 2))  # 48 for 384 input
    w = 2 * (int(width) // (pipe.vae_scale_factor * 2))   # 48 for 384 input
    if not pack:
        # Unpacked 4D shape: (B, num_channels*4, h//2, w//2) = (1, 128, 24, 24)
        shape = (batch_size, num_channels_latents * 4, h // 2, w // 2)
    else:
        # Packed 3D shape: (B, (h//2)*(w//2), num_channels*4) = (1, 576, 128)
        shape = (batch_size, (h // 2) * (w // 2), num_channels_latents * 4)
    return shape

# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def get_flux2_latent_image_ids(pipe, generator_kwargs, batch_size=1):
    """Get latent image IDs for Flux2 pipeline."""
    device, dtype = pipe.device, pipe.dtype
    # get_flux2_latent_shape with pack=False returns (B, 128, 24, 24)
    b, c, h, w = get_flux2_latent_shape(pipe, generator_kwargs, batch_size, pack=False)
    # _prepare_latent_ids expects the same 4D shape
    latent_image_ids = pipe._prepare_latent_ids(
        torch.zeros(batch_size, c, h, w, device=device, dtype=dtype)
    )
    return latent_image_ids

# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def get_flux2_guidance(pipe, generator_kwargs):
    """Get guidance embedding for Flux2 pipeline."""
    device = pipe.device
    guidance = torch.tensor([generator_kwargs["guidance_scale"]], device=device)
    return guidance

# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def run_flux2_forward(pipe, generator_kwargs, latents, t, forward_kwargs):
    """Run forward pass for Flux2 transformer."""
    # Ensure latents are packed 3D (B, seq, channels)
    # FLUX2 pipeline returns packed latents, but when stop_i=0, we might get unpacked
    if latents.ndim == 4:
        latents = pipe._pack_latents(latents)

    latent_image_ids = get_flux2_latent_image_ids(pipe, generator_kwargs)
    guidance = get_flux2_guidance(pipe, generator_kwargs)
    forward_kwargs = {
        "hidden_states": latents,
        "timestep": (t / 1000),
        "guidance": guidance,
        "img_ids": latent_image_ids,
        "return_dict": False,
        **forward_kwargs
    }
    model_pred = pipe.transformer(**forward_kwargs)[0]
    return model_pred, forward_kwargs

def get_flux_latent_image_ids(pipe, generator_kwargs, batch_size=1):
    device, dtype = pipe.device, pipe.dtype
    b, c, h, w = get_flux_latent_shape(pipe, generator_kwargs, batch_size, pack=False)
    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size,
        h // 2, 
        w // 2,
        device,
        dtype,
    )
    return latent_image_ids

def get_flux_guidance(pipe, generator_kwargs):
    if pipe.transformer.config.guidance_embeds:
        device = pipe.device
        guidance = torch.tensor([generator_kwargs["guidance_scale"]]).to(device)
        return guidance
    else:
        return None

def run_flux_forward(pipe, generator_kwargs, latents, t, forward_kwargs):
    latent_image_ids = get_flux_latent_image_ids(pipe, generator_kwargs)
    guidance = get_flux_guidance(pipe, generator_kwargs)
    forward_kwargs = {
        "hidden_states": latents,
        "timestep": (t / 1000),
        "guidance": guidance,
        "img_ids": latent_image_ids,
        "return_dict": False,
        **forward_kwargs
    }
    model_pred = pipe.transformer(**forward_kwargs)[0]
    return model_pred, forward_kwargs

# ===========================
#     SD Pipeline Helpers
# ===========================
def get_sd_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    backbone = get_backbone(pipe)
    num_channels_latents = backbone.config.in_channels
    height, width = generator_kwargs["height"], generator_kwargs["width"]
    height = int(height) // pipe.vae_scale_factor
    width = int(width) // pipe.vae_scale_factor
    shape = (batch_size, num_channels_latents, height, width)
    return shape

def encode_sd_text(pipe, prompts, keys, device):
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "encoder_hidden_states": prompt_embeds,
        }
    return prompt_kwargs

def run_sd_forward(pipe, generator_kwargs, latents, t, forward_kwargs, sample_key="sample"):
    backbone = get_backbone(pipe)
    b = forward_kwargs["encoder_hidden_states"].shape[0] // 2
    latents = latents.expand(b * 2, *latents.shape[1:])
    t = t.expand(latents.shape[0])
    forward_kwargs = {
        sample_key: latents,
        "timestep": t,
        "return_dict": False,
        **forward_kwargs
    }
    noise_pred = backbone(**forward_kwargs)[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    model_pred = noise_pred_uncond + generator_kwargs["guidance_scale"] * (noise_pred_text - noise_pred_uncond)
    forward_kwargs["hidden_states"] = forward_kwargs.pop(sample_key)[:b]
    return model_pred, forward_kwargs

# ===========================
#    Sana Pipeline Helpers
# ===========================
def get_sana_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    return get_sd_latent_shape(pipe, generator_kwargs, batch_size, pack)

def encode_sana_text(pipe, prompts, keys, device):
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_attention_mask
        }
    return prompt_kwargs

def run_sana_forward(pipe, generator_kwargs, latents, t, forward_kwargs):
    return run_sd_forward(pipe, generator_kwargs, latents, t, forward_kwargs, sample_key="hidden_states")

# ===========================
#    Qwen Pipeline Helpers
# ===========================
# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def get_qwen_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    """
    Get latent shape for Qwen-Image pipeline.

    Qwen-Image uses:
    - VAE z_dim = 16 (latent channels)
    - Transformer in_channels = 64 (patchified: 16 * 4 patches)
    - vae_scale_factor = 8

    IMPORTANT: Qwen pipeline returns and expects PACKED 3D latents (batch, seq_len, 64)
    when using output_type='latent' or passing latents parameter.

    When pack=False: returns 4D shape (batch, channels, height, width) - unpacked format
    When pack=True: returns 3D shape (batch, height*width/4, 64) - packed format for pipeline
    """
    # Use VAE z_dim for latent channels, NOT transformer.config.in_channels
    if hasattr(pipe, 'vae') and hasattr(pipe.vae.config, 'z_dim'):
        num_channels_latents = pipe.vae.config.z_dim  # Should be 16
    else:
        # Fallback: use transformer out_channels which matches VAE z_dim
        backbone = get_backbone(pipe)
        if hasattr(backbone.config, "out_channels"):
            num_channels_latents = backbone.config.out_channels  # 16
        else:
            num_channels_latents = 16  # Qwen-Image default

    height, width = generator_kwargs["height"], generator_kwargs["width"]
    scale_factor = getattr(pipe, "vae_scale_factor", 8)

    # Qwen uses: 2 * (int(height) // (vae_scale_factor * 2))
    latent_height = 2 * (int(height) // (scale_factor * 2))
    latent_width = 2 * (int(width) // (scale_factor * 2))

    if pack:
        # Packed 3D shape: (batch, seq_len, in_channels)
        # seq_len = (h/2) * (w/2), in_channels = 16 * 4 = 64
        seq_len = (latent_height // 2) * (latent_width // 2)
        in_channels = num_channels_latents * 4  # 16 * 4 = 64
        shape = (batch_size, seq_len, in_channels)
    else:
        # Unpacked 4D shape
        shape = (batch_size, num_channels_latents, latent_height, latent_width)
    return shape

# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def encode_qwen_text(pipe, prompts, keys, device):
    """
    Encode text for Qwen-Image pipeline.

    Qwen-Image uses a VLM-based text encoder. The encode_prompt method returns:
    - prompt_embeds: (batch, seq_len, hidden_size) e.g., (1, 8, 3584)
    - prompt_embeds_mask: (batch, seq_len) e.g., (1, 8)

    For LoRA training, we need to encode text and pass embeddings to the transformer.

    NOTE: For Qwen, we store positive and negative embeddings SEPARATELY (not concatenated)
    because the pipeline requires masks to match embedding lengths exactly.
    """
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        # Encode positive prompt
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
        )

        # For CFG, encode negative prompt (empty string)
        negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
            prompt=" ",  # Qwen uses space as empty prompt
            device=device,
            num_images_per_prompt=1,
        )

        # Store separately for Qwen (pipeline requires unpadded masks)
        # Use special keys that run_pipe will recognize for Qwen
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "encoder_hidden_states": prompt_embeds,  # positive only
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "negative_encoder_hidden_states": negative_prompt_embeds,
            "negative_encoder_hidden_states_mask": negative_prompt_embeds_mask,
        }

    return prompt_kwargs


# [DesignLoRA] FLUX.2 / Qwen-Image generator support (not in upstream)
def run_qwen_forward(pipe, generator_kwargs, latents, t, forward_kwargs):
    """
    Run forward pass for Qwen-Image transformer.

    Qwen-Image transformer expects:
    - hidden_states: packed latents (batch, seq_len, in_channels=64)
    - encoder_hidden_states: text embeddings (batch, text_seq_len, hidden_size)
    - encoder_hidden_states_mask: attention mask for text
    - timestep: diffusion timestep

    NOTE: Latents can be in different formats:
    - 3D packed: (batch, seq_len, 64) - from pipeline output_type='latent'
    - 4D unpacked: (batch, 16, height, width)
    - 5D: (batch, 1, 16, height, width)
    """
    backbone = get_backbone(pipe)

    # Get pixel-space dimensions from generator_kwargs
    img_height = generator_kwargs["height"]
    img_width = generator_kwargs["width"]
    scale_factor = getattr(pipe, "vae_scale_factor", 8)

    # Latent space dimensions (before packing)
    latent_height = 2 * (int(img_height) // (scale_factor * 2))
    latent_width = 2 * (int(img_width) // (scale_factor * 2))

    # Handle different latent dimensions
    if latents.dim() == 3:
        batch_size = latents.shape[0]
        packed_latents = latents.contiguous().to(pipe.device, pipe.dtype)
    elif latents.dim() == 5:
        batch_size, _, num_channels, height, width = latents.shape
        latents_4d = latents.squeeze(1).contiguous().to(pipe.device, pipe.dtype)
        packed_latents = pipe._pack_latents(latents_4d, batch_size, num_channels, height, width)
    elif latents.dim() == 4:
        batch_size, num_channels, height, width = latents.shape
        latents_4d = latents.contiguous().to(pipe.device, pipe.dtype)
        packed_latents = pipe._pack_latents(latents_4d, batch_size, num_channels, height, width)
    else:
        raise ValueError(f"Expected 3D, 4D, or 5D latents, got {latents.dim()}D: {latents.shape}")

    # Check for encoder_hidden_states in forward_kwargs
    encoder_hidden_states = forward_kwargs.get("encoder_hidden_states")
    encoder_hidden_states_mask = forward_kwargs.get("encoder_hidden_states_mask")

    # Expand for CFG if encoder_hidden_states has batch size 2 (neg + pos)
    if encoder_hidden_states is not None and encoder_hidden_states.shape[0] > batch_size:
        packed_latents = torch.cat([packed_latents] * 2)
        t = t.expand(packed_latents.shape[0])

    # Prepare forward kwargs for Qwen transformer
    transformer_kwargs = {
        "hidden_states": packed_latents,
        "timestep": t,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "return_dict": False,
    }

    # Add img_shapes and txt_seq_lens if needed
    if encoder_hidden_states is not None:
        txt_seq_len = encoder_hidden_states.shape[1]
        current_batch = packed_latents.shape[0]

        img_h = latent_height // 2
        img_w = latent_width // 2
        transformer_kwargs["img_shapes"] = [[(1, img_h, img_w)]] * current_batch

        transformer_kwargs["txt_seq_lens"] = torch.tensor(
            [txt_seq_len] * current_batch,
            device=packed_latents.device,
            dtype=torch.long
        )

    # Forward pass
    model_pred = backbone(**transformer_kwargs)[0]

    # Unpack output to (B, C, H, W) format
    model_pred = pipe._unpack_latents(model_pred, img_height, img_width, scale_factor)
    if model_pred.dim() == 5:
        model_pred = model_pred.squeeze(2)

    # Apply CFG if we had expanded latents
    if model_pred.shape[0] == 2 * batch_size:
        noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        guidance_scale = generator_kwargs.get("guidance_scale", 4.0)
        model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Store the original latent shape for loss computation (4D)
    if latents.dim() == 3:
        latents_4d = pipe._unpack_latents(latents, img_height, img_width, scale_factor)
        if latents_4d.dim() == 5:
            latents_4d = latents_4d.squeeze(2)
    elif latents.dim() == 5:
        latents_4d = latents.squeeze(1)
    else:
        latents_4d = latents
    transformer_kwargs["hidden_states"] = latents_4d

    return model_pred, transformer_kwargs