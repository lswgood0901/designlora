# `dual_process/` — provenance and DesignLoRA changes

This package is adapted from **Dual-Process Image Generation** (Luo, Granskog,
Holynski, Darrell — ICCV 2025), the VLM-guided gradient-distillation method that
DesignLoRA builds on.

- Upstream repository: https://github.com/g-luo/dual_process
- Project page: https://dual-process.github.io
- Upstream files are kept as close to the original as practical. Every DesignLoRA
  edit inside these files is tagged with a `# [DesignLoRA]` comment. This file is
  the authoritative summary of what changed.

DesignLoRA extends the upstream VQA-gradient loop (question-conditioned only) into
a **dual-signal** loop that also conditions on accumulated selection history via a
Bradley–Terry (BTL) preference model, and adds Expected-Improvement candidate
sampling. See the top-level `README.md` and `PAPER_VS_CODE.md`.

## Per-file status

| File | Status | Notes |
|------|--------|-------|
| `dig_viz.py` | **unchanged** | Upstream verbatim (in-memory image grids / loss plots). |
| `gpt_helpers.py` | **unchanged** | Upstream verbatim (OpenAI message helpers). |
| `dig_helpers.py` | **modified** | +FLUX.2 and Qwen-Image **generator** support; `get_pipe_cls` dispatch. |
| `dig_operators.py` | **modified** | +`create_multi_image_vlm_prompt` for the dual-signal prompt. |
| `dig_pipeline.py` | **modified** | `create_edit` / `inner_loop` extended for the dual (question + preference) prompt path. |
| `llm_surrogate.py` | **rewritten** | DesignLoRA's Bradley–Terry preference surrogate. |
| `preference_model.py` | **removed** | Dead duplicate of `llm_surrogate.py` (imported nowhere). |

## DesignLoRA additions (tagged `[DesignLoRA]` in the code)

- **`dig_helpers.py`** — image-generator backends absent from upstream (which shipped
  Stable Diffusion / Sana / FLUX.1):
  `encode_flux2_text`, `get_flux2_latent_shape`, `get_flux2_latent_image_ids`,
  `get_flux2_guidance`, `run_flux2_forward` (FLUX.2), and
  `get_qwen_latent_shape`, `encode_qwen_text`, `run_qwen_forward` (Qwen-Image).
  `get_pipe_cls` gained `flux2` / `qwen` branches.
- **`dig_operators.py`** — `create_multi_image_vlm_prompt`: builds the
  preference-aware prompt that shows the VLM the four reference images (A/B/C/D)
  with their BTL goodness scores plus the candidate (paper §3.3, Appendix A).
- **`llm_surrogate.py`** — rewritten into the BTL preference model:
  `bt_update`, `bt_score`, `bt_mle`, `bt_clear`, `rank_images`. The goodness scores
  are injected into the preference-aware VLM prompt as numeric context.

Note: the upstream file already supported the **Qwen VLM** (`preprocess_qwen_image`,
`loss_vlm_multiqa`, etc. in `dig_operators.py`); those are *not* DesignLoRA additions.
DesignLoRA adds Qwen/**FLUX.2 as image generators**, which is separate.

## Dead code removed during cleanup

These were experimental DesignLoRA-era code paths that the live server never called,
removed for clarity (they are documented here so their absence is intentional, not lossy):

- `dig_pipeline.py`: `one_step_bt_update`, `train_step_vqa`, `_get_latents_for_prompt`
  (a Bradley–Terry *gradient*-loss training path — **not** the method in the paper,
  whose loss is the dual VQA objective `L_VQA`, Eq. 5).
- `dig_operators.py`: `loss_bt_vlm` (the BT logistic loss used only by the above).
- `preference_model.py`, `llm_surrogate_backup.py`, `llm_surrogate_clean.py` (duplicates).
- top-level `ei_bt/` package + `configs/ei_bt.yaml` (a standalone PBO/EI experiment,
  imported nowhere; DesignLoRA's live EI is implemented in `designlora_server.py`).

The live training loss remains upstream's dual VQA objective
(`dig_pipeline.inner_loop` → `dig_operators.loss_vlm_multiqa`), consistent with the paper.
