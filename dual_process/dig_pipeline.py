# -----------------------------------------------------------------------------
# Adapted from Dual-Process Image Generation (g-luo/dual_process, ICCV 2025).
# DesignLoRA additions/changes are tagged "[DesignLoRA]"; see
# dual_process/UPSTREAM_CHANGES.md for the full list.
# -----------------------------------------------------------------------------
import bitsandbytes as bnb
import json
from IPython.display import display, clear_output
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
from PIL import Image
import torch
from tqdm import tqdm

from dual_process import dig_helpers, dig_operators, dig_viz, llm_surrogate

# ===========================
#        Create Edit
# ===========================
@torch.no_grad()
def create_vlm_edit(edit, vlm, vlm_processor, config, qa_pairs):
    # Add space to question postfix if it doesn't already have one
    question_postfix = config.get("question_postfix", "")
    if question_postfix and question_postfix[0] != " ":
        question_postfix = " " + question_postfix
    # Iterate over qa_pairs
    guidance_prompts = []
    for qa in qa_pairs:
        question, answer = qa["question"], qa["answer"]
        question = question + question_postfix
        # Process ref_images
        ref_images = qa.get("ref_images", [])
        overlay_mode = qa.get("overlay_mode")
        candidate_image = qa.get("candidate_image")  # For multi-image mode
        
        if not ref_images:
            ref_images = []
        if type(ref_images) is not list:
            ref_images = [ref_images]
        ref_images = [Image.open(image) if type(image) is str else image for image in ref_images]
        ref_images = [image.convert("RGB") for image in ref_images]
        # Use custom image_dim from qa if provided (for VRAM reduction), otherwise use generator dimensions
        image_dim = qa.get("image_dim", (config["generator_kwargs"]["width"], config["generator_kwargs"]["height"]))
        
        # Use multi-image prompt if ref_images are provided and we want multi-image mode
        if len(ref_images) > 0 and config.get("use_multi_image_vlm", False):
            guidance_prompt = dig_operators.create_multi_image_vlm_prompt(
                vlm_processor, 
                question,
                answer,
                vlm_template=config["vlm_template"],
                ref_images=ref_images,
                candidate_image=candidate_image,
                image_dim=image_dim
            )
        else:
            # Create guidance_prompt for vlm (original single-image mode)
            guidance_prompt = dig_operators.create_vlm_prompt(
                vlm_processor, 
                question,
                answer,
                vlm_template=config["vlm_template"],
                ref_images=ref_images,
                image_dim=image_dim,
                overlay_mode=overlay_mode
            )
        guidance_prompts.append(guidance_prompt)
    # carry generation kwargs for VLM free-form into the edit
    gen_kwargs = config.get("vlm_generate_kwargs", {})
    if isinstance(gen_kwargs, DictConfig):
        gen_kwargs = OmegaConf.to_container(gen_kwargs, resolve=True)
    edit["vlm_generate_kwargs"] = gen_kwargs or {}
    edit["vlm"] = vlm
    edit["vlm_processor"] = vlm_processor
    edit["guidance_prompts"] = guidance_prompts
    edit["loss_fn"] = "loss_vlm_multiqa"
    return edit

@torch.no_grad()
def create_pipe_edit(edit, pipe, prompt, prompt_keys=["target"]):
    if type(prompt) is not list:
        prompt = [prompt]
    pipe_cls = dig_helpers.get_pipe_cls(pipe)
    encode_text_fn = getattr(dig_helpers, f"encode_{pipe_cls}_text")
    prompt_kwargs = encode_text_fn(pipe, prompt, prompt_keys, pipe.device)
    edit = {**edit, **prompt_kwargs}
    return edit

def create_edit(pipe, vlm, vlm_processor, config, qa_pairs, prompt):
    edit = config.get("edit_kwargs", {})
    if isinstance(edit, DictConfig):
        edit = OmegaConf.to_container(edit, resolve=True)
    edit = create_vlm_edit(edit, vlm, vlm_processor, config, qa_pairs)
    edit = create_pipe_edit(edit, pipe, prompt)
    return edit

# ===========================
#         Checkpoint
# ===========================
@torch.no_grad()
def run_vlm(pipe, edit, pred_x0, max_new_tokens=10):
    vlm, vlm_processor = edit["vlm"], edit["vlm_processor"]

    # Synchronize CUDA to ensure previous operations are complete
    # This helps catch any latent errors from distributed GPU operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    try:
        input_ids, pixel_values, vlm_kwargs, optimize_mask = dig_operators.get_vlm_args(pipe, edit, pred_x0=pred_x0)

        # Check for NaN/Inf in pixel values (can happen with multi-GPU distribution)
        if pixel_values is not None and torch.is_tensor(pixel_values):
            if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                print("WARNING: pixel_values contains NaN or Inf, clamping values")
                pixel_values = torch.nan_to_num(pixel_values, nan=0.0, posinf=1.0, neginf=-1.0)
                pixel_values = pixel_values.clamp(-10.0, 10.0)

        input_ids = input_ids[:, ~optimize_mask]
        # merge per-config generation kwargs (sampling, penalties, etc.)
        gen_kwargs = edit.get("vlm_generate_kwargs", {}) or {}
        # explicit call arg wins over config if both provided
        if "max_new_tokens" not in gen_kwargs and max_new_tokens is not None and max_new_tokens >= 0:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        # Add do_sample=False to avoid probability sampling issues
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False

        outputs = vlm.generate(input_ids=input_ids, pixel_values=pixel_values, **vlm_kwargs, **gen_kwargs)
        answer = vlm_processor.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return answer
    except RuntimeError as e:
        if "CUDA" in str(e) or "probability" in str(e):
            print(f"WARNING: VLM generation failed with CUDA error: {e}")
            # Reset CUDA state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ""
        raise

def run_eval(pipe, generator_kwargs, prompt, eval_weights, eval_n, eval_seed, eval_noise=None, low_memory=False, use_llm_surrogate=True):
    if use_llm_surrogate:
        eval_images = []
        for weight in eval_weights:
            generator = torch.Generator().manual_seed(eval_seed)
            eval_kwargs = {
                "pipe": pipe,
                "prompt": prompt,
                "generator_kwargs": generator_kwargs,
                "generator": generator
            }
            with dig_helpers.LoraManager(pipe, enter_weights=weight):
                if low_memory:
                    images = []
                    for i in range(eval_n):
                        latents = eval_noise[i][None, ...] if eval_noise is not None else None
                        image = dig_helpers.run_pipe(num_images_per_prompt=1, latents=latents, **eval_kwargs)[0]
                        images.append(image)
                else:
                    images = dig_helpers.run_pipe(num_images_per_prompt=eval_n, latents=eval_noise, **eval_kwargs)
            eval_images.append(images)
        eval_images = [list(row) for row in zip(*eval_images)]
        eval_images = sum(eval_images, [])
        return eval_images
    generator = torch.Generator().manual_seed(eval_seed)
    images = dig_helpers.run_pipe(
        pipe=pipe,
        prompt=prompt,
        generator_kwargs=generator_kwargs,
        generator=generator,
        num_images_per_prompt=eval_n * 2,     # 충분히 생성
        latents=eval_noise,
    )

    # ---------- 2단계: LLM Surrogate로 평가 → 상위 eval_n 장 ----------
    top_imgs, probs = llm_surrogate.rank_images(prompt, images, top_k=eval_n)

    # ---------- 3단계: BT 업데이트 ----------
    best_id = id(top_imgs[0])
    for img in top_imgs[1:]:
        llm_surrogate.bt_update([(best_id, id(img))])
    return top_imgs

@torch.no_grad()
def save_lora_results(pipe, generator_kwargs, prompt, save_folder, config, o, edit, results):
    if not save_folder:
        return
    o = str(o).zfill(6)
    if config["opt_kwargs"].get("save_weights", True):
        dig_helpers.save_weights(pipe, f"{save_folder}/lora_o-{str(o).zfill(6)}")
    # Visualize eval seeds
    result_images = []
    eval_n = config["eval_kwargs"]["eval_n"]
    eval_images = run_eval(
        pipe, 
        generator_kwargs, 
        prompt, 
        **config["eval_kwargs"]
    )
    eval_grid = dig_viz.view_images(eval_images, num_rows=eval_n)
    eval_grid.save(f"{save_folder}/eval_i-{0}_o-{o}.png")
    result_images.extend(eval_images)
    # ----------- BT‑based score logging -----------
    # Compute Bradley‑Terry probabilities (via llm_surrogate) for each eval image
    bt_scores = [llm_surrogate.bt_score(id(img)) for img in eval_images]

    # Store into results["metas_probs"] under current iteration key
    if results is not None:
        results.setdefault("metas_probs", {})
        results["metas_probs"][f"lora_o-{o}"] = bt_scores
    # Save vlm predictions
    if results and "vlm" in edit["loss_fn"]:
        loss_fn = getattr(dig_operators, edit["loss_fn"])
        metas = [loss_fn(pipe=pipe, edit=edit, pred_x0=image)[1] for image in result_images]
        for k in metas[0].keys():
            scores = [m[k] for m in metas]
            scores = [s.tolist() if torch.is_tensor(s) else s for s in scores]
            results[f"metas_{k}"] = results.get(f"metas_{k}", {})
            results[f"metas_{k}"][f"lora_o-{o}"] = scores
    # Save losses
    if results:
        dig_viz.plot_losses(results["losses"], "Step", "Loss", "Loss").save(f"{save_folder}/losses.png")
        results = {k: v for k, v in results.items() if not torch.is_tensor(v)}
        json.dump(results, open(f"{save_folder}/results.json", "w"))

@torch.no_grad()
def visualize_vanilla_results(pipe, generator_kwargs, prompt, edit, grid_size, eval_seed, eval_weight, max_new_tokens=-1, text_height=50, text_scale=1.5, eval_noise=None):
    eval_n = grid_size[1]
    grid = []
    for r in range(grid_size[0]):
        eval_images = run_eval(pipe, generator_kwargs, prompt, [eval_weight], eval_n, eval_seed + r, eval_noise=eval_noise)
        metas = None
        if "vlm" in edit["loss_fn"]:
            # Generate with vlm
            if max_new_tokens >= 1:
                metas = [run_vlm(pipe, edit, pred_x0=image, max_new_tokens=max_new_tokens) for image in eval_images]
            else:
                # Rate with vlm
                loss_fn = getattr(dig_operators, edit["loss_fn"])
                format_score = lambda scores: [f"{round(s, 2)}" for s in scores]
                metas = [loss_fn(pipe=pipe, edit=edit, pred_x0=image)[1] for image in eval_images]
                metas = [f"{x['preds']}\n{format_score(x['probs'].tolist())}" for x in metas]
            # Add overlay
            guidance_prompt = edit["guidance_prompts"][0]
            ref_images = guidance_prompt.get("ref_images", [])
            eval_images = [dig_operators.create_visual_prompt(pipe, guidance_prompt, eval_image, ref_images, mode="pil") for eval_image in eval_images]
        row = dig_viz.view_images(eval_images, 1)
        if metas is not None:
            row_scores = dig_viz.view_images([dig_viz.display_text(row.size[0] // grid_size[0], text_height, meta, scale=text_scale) for meta in metas], offset_ratio=0)
            row_scores = dig_viz.resize_to_side(row_scores, row.size[0], fn="max")
            grid.append(row_scores)
        grid.append(row)
    grid = dig_viz.stack_images(grid)
    return grid

def interactive_vanilla_results(pipe, generator_kwargs, prompt, edit, save_folder, o, train_weight, eval_noise, text_height=100, text_scale=3):
    base_image = visualize_vanilla_results(pipe, generator_kwargs, prompt, edit, (1, 1), eval_seed=0, eval_weight=0, eval_noise=eval_noise, text_height=text_height, text_scale=text_scale)
    lora_image = visualize_vanilla_results(pipe, generator_kwargs, prompt, edit, (1, 1), eval_seed=0, eval_weight=train_weight, eval_noise=eval_noise, text_height=text_height, text_scale=text_scale)
    image = dig_viz.view_images([base_image, lora_image], 1)
    title = f"iter: {o}"
    title = dig_viz.display_text(image.size[0], text_height, title, scale=text_scale)
    image = dig_viz.stack_images([title, image])
    if save_folder:
        if not os.path.exists(f"{save_folder}/readout"):
            os.makedirs(f"{save_folder}/readout", exist_ok="True")
        image.save(f"{save_folder}/readout/viz-{str(o).zfill(6)}.png")
    clear_output(wait=True)
    display(image)

# ===========================
#         Optimize
# ===========================
class IterationChecker:
    def __init__(self):
        self.break_iter = False

def inner_loop(pipe, edit, generator_kwargs, generator, train_weight, init_noise, o):
    # ===========================
    #      Synthetic Data
    # ===========================
    pipe_cls = dig_helpers.get_pipe_cls(pipe)
    with dig_helpers.LoraManager(pipe, enter_weights=train_weight):
        with torch.no_grad():
            if edit.get("subsample", True):
                # Generate and early stop
                # timestep is therefore subsampled
                pipe.scheduler_config["num_train_timesteps"] = generator_kwargs["num_inference_steps"]
                i, t = dig_helpers.get_timestep(pipe)
                latents = dig_helpers.run_pipe(
                    pipe,
                    prompt_kwargs=edit["target_prompt_kwargs"],
                    generator_kwargs=generator_kwargs,
                    generator=generator,
                    latents=init_noise,
                    output_type="latent",
                    stop_i=i,
                )
            else:
                # Generate fully, then add noise
                # at random timestep in 0 to 1000
                pipe.scheduler_config["num_train_timesteps"] = pipe.scheduler.num_train_timesteps
                i, t = dig_helpers.get_timestep(pipe)
                latents = dig_helpers.run_pipe(
                    pipe,
                    prompt_kwargs=edit["target_prompt_kwargs"],
                    generator_kwargs=generator_kwargs,
                    generator=generator,
                    latents=init_noise,
                    output_type="latent"
                )
                random_noise = torch.randn_like(latents)
                if hasattr(pipe.scheduler, "add_noise"):
                    latents = pipe.scheduler.add_noise(latents, random_noise, t)
                elif hasattr(pipe.scheduler, "scale_noise"):
                    latents = pipe.scheduler.scale_noise(latents, t, random_noise)
                else:
                    raise NotImplementedError
            latents = latents.to(pipe.device)
            latents = latents.to(pipe.dtype)
    # ===========================
    #       Forward Pass
    # ===========================
    with dig_helpers.LoraManager(pipe, enter_weights=train_weight):
        forward_fn = getattr(dig_helpers, f"run_{pipe_cls}_forward")
        model_pred, forward_kwargs = forward_fn(
            pipe,
            generator_kwargs, 
            latents, 
            t, 
            edit["target_prompt_kwargs"]
        )
    # ===========================
    #            Loss
    # ===========================
    loss_fn = getattr(dig_operators, edit["loss_fn"])
    loss, meta = loss_fn(
        pipe=pipe, 
        edit=edit, 
        generator_kwargs=generator_kwargs, 
        forward_kwargs=forward_kwargs, 
        model_pred=model_pred, 
        i=i, 
        t=t, 
        o=o
    )
    
    # Apply loss multiplier if provided (for directional training like designlora)
    if "loss_multiplier" in edit:
        loss = loss * edit["loss_multiplier"]
    
    return loss

def get_optimizer(params, **opt_kwargs):
    opt_cls = opt_kwargs.get("opt_cls", "torch.optim.AdamW")
    opt_cls = eval(opt_cls)
    optimizer = opt_cls(params=params, **opt_kwargs.get("optimizer_kwargs"))
    return optimizer

def optimize(pipe, edit, opt_kwargs, generator_kwargs, params, prompt=None, save_folder=None, config=None, eval_interactive=None, iteration_checker=None):
    # Configure hparams
    num_opt_steps = opt_kwargs["num_opt_steps"]
    num_accum_steps = opt_kwargs.get("num_accum_steps", 1)
    train_weight = opt_kwargs.get("train_weight", 1)
    train_n = opt_kwargs.get("train_n", num_opt_steps)
    train_random = opt_kwargs.get("train_random", False)
    eval_every_n = opt_kwargs.get("eval_every_n", None)
    eval_interactive = eval_interactive or opt_kwargs.get("eval_interactive")
    # Configure training
    generator = torch.Generator().manual_seed(opt_kwargs.get("train_seed", 100))
    optimizer = get_optimizer(params, **opt_kwargs)
    results = {"losses": []}
    pbar = tqdm(total=num_opt_steps)
    # Seed training noise
    pipe_cls = dig_helpers.get_pipe_cls(pipe)
    get_latent_shape_fn = getattr(dig_helpers, f"get_{pipe_cls}_latent_shape")
    # FLUX2 expects unpacked 4D latents as input (it packs internally), unlike FLUX.1 which accepts packed 3D
    pack_latents = False if pipe_cls == "flux2" else True
    latent_shape = get_latent_shape_fn(pipe, generator_kwargs, pack=pack_latents)
    train_noise = [torch.randn(latent_shape, generator=generator) for _ in range(train_n)]
    train_noise = torch.vstack(train_noise)
    train_noise = train_noise.to(pipe.dtype)
    for o in range(num_opt_steps):
        # Empty cache
        torch.cuda.empty_cache()
        # Early stop for interactive gradio app
        if iteration_checker and iteration_checker.break_iter:
            save_lora_results(pipe, generator_kwargs, prompt, save_folder, config, o, edit, results)
            break
        # Select either ordered or random seed
        train_idx = np.random.choice(train_noise.shape[0]) if train_random else o % train_noise.shape[0]
        init_noise = train_noise[train_idx][None, ...]
        loss = inner_loop(pipe, edit, generator_kwargs, generator, train_weight, init_noise, o)
        results["losses"].append(loss.item())
        # ===========================
        #      Optimizer Step
        # ===========================
        loss = loss / num_accum_steps
        loss.backward()
        if (o + 1) % num_accum_steps == 0:
            # Gradient clipping
            max_param_norm = opt_kwargs.get("max_param_norm", 1.0)
            for p in params:
                torch.nn.utils.clip_grad_norm_(p["params"], max_param_norm)
            optimizer.step()
            optimizer.zero_grad()
        # ===========================
        #          Logging
        # ===========================
        postfix = {"loss": results["losses"][-1]}
        pbar.set_postfix(**postfix)
        pbar.update(1)

        if eval_interactive:
            if o % eval_interactive == 0:
                interactive_vanilla_results(pipe, generator_kwargs, prompt, edit, save_folder, o, train_weight, init_noise)

        if eval_every_n:
            if o > 0 and o % eval_every_n == 0:
                save_lora_results(pipe, generator_kwargs, prompt, save_folder, config, o, edit, results)

    if eval_every_n:
        save_lora_results(pipe, generator_kwargs, prompt, save_folder, config, o, edit, results)

    return results
