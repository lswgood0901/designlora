# DesignLoRA

**Interactive Fine-Tuning of Text-to-Image Models with Designer-in-the-Loop Feedback**

Seung Won Lee, Yejin Yun, Kyung Hoon Hyun · Hanyang University

DesignLoRA is a designer-in-the-loop method that personalizes a text-to-image
generator to a designer's **context-specific preferences** in near-real-time.
At each round the designer asks a natural-language question and picks a preferred
image; the system combines two complementary VLM signals — one **question-conditioned**
(explicit intent) and one **preference-aware** (implicit intent, from Bradley–Terry
scores over the selection history) — into a VQA gradient that updates a LoRA on the
generator, then proposes the next candidates by **Expected Improvement (EI)**.

It builds on [Dual-Process Image Generation](https://github.com/g-luo/dual_process)
(Luo et al., ICCV 2025); see [`dual_process/UPSTREAM_CHANGES.md`](dual_process/UPSTREAM_CHANGES.md)
for exactly what DesignLoRA changes, and [`PAPER_VS_CODE.md`](PAPER_VS_CODE.md) for how
the code maps to the paper.

---

## Method at a glance

One interaction round (`POST /interact/designlora` or `POST /finetune`):

1. **Generate** a 4-image panel with the current LoRA-adapted generator.
2. **Select** — the designer picks a winner and (optionally) asks a question.
3. **Preference update** — a Bradley–Terry (BTL) model updates goodness scores from
   the winner ≻ losers comparisons (`dual_process/llm_surrogate.py`).
4. **Dual-prompt VLM evaluation** — the VLM scores each candidate two ways:
   `p_question = P("Yes" | candidate, reference, question)` and
   `p_pref = P("Yes" | candidate, {reference images + BTL scores})`.
5. **LoRA adaptation** — minimize the weighted VQA loss
   `L = -[w_q·log p_question + w_p·log p_pref]` (`w_q = w_p = 0.5`).
6. **EI sampling** — sample `k` seeds, run `n` VLM evaluations each, and keep the
   top candidates by Expected Improvement (no Gaussian process).

## Setup

```bash
conda env create -f environment.yaml
conda activate designlora
```

Point the server at your local model weights (defaults to `/data1/models`):

```bash
export DESIGNLORA_MODELS_DIR=/path/to/models
```

Every model path in `configs/` resolves as
`${DESIGNLORA_MODELS_DIR}/<ModelName>`, so no machine-specific paths are baked in.

## Run

```bash
conda activate designlora
python designlora_server.py          # FastAPI + uvicorn on http://0.0.0.0:8002
```

Check it is up: `curl http://localhost:8002/health`.

## Models

DesignLoRA is generator- and VLM-agnostic. Select a generator with the `imageModel`
request field and a VLM in `configs/base.yaml` (`configs/vlm/*.yaml`, default `idefics2`).

| `imageModel` | Config | Notes | Requires |
|--------------|--------|-------|----------|
| `schnell` (default) | `configs/pipe/schnell.yaml` | FLUX.1-schnell — **paper's primary system** | diffusers ≥ 0.32 |
| `dev` | `configs/pipe/dev.yaml` | FLUX.1-dev | diffusers ≥ 0.32 |
| `flux2` | `configs/pipe/flux2.yaml` | FLUX.2 [dev] (Appendix B) | diffusers ≥ 0.36 |
| `flux2_klein` | `configs/pipe/flux2_klein.yaml` | FLUX.2 [klein] 9B, open-weight, step-distilled | diffusers ≥ 0.37 |
| `qwen_image` | `configs/pipe/qwen_image.yaml` | Qwen-Image (Appendix B) | diffusers ≥ 0.34 |

VLMs (`configs/vlm/`): `idefics2` (default, paper), `qwenvl`, plus experimental
`gemma3` / `llava` / `pixtral`.

**Newer generators / GPUs.** FLUX.2 support lives lazily in
`dual_process/dig_helpers.py`, so the base environment imports fine — you only need
the newer `diffusers` when you actually select `flux2*`. For recent GPUs
(e.g. NVIDIA Blackwell / RTX 50xx) install a CUDA-matched build, e.g.:

```bash
pip install --upgrade "diffusers>=0.37" "transformers>=5"
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## API

Base URL `http://localhost:8002`. Models auto-load on first request (calling `/setup`
is optional).

| Endpoint | Purpose |
|----------|---------|
| `POST /interact/designlora` | One interaction round on the active session (select + question → train + resample). |
| `POST /finetune` | Same loop with named, versioned checkpoints (`checkpoint_id`). |
| `POST /generate_with_lora` | Generate from a checkpoint (no training). |
| `GET  /checkpoints/{userID}` | List a user's checkpoints. |
| `POST /setup`, `POST /generate` | Explicit session setup / plain generation. |
| `GET  /health` | Liveness probe. |

> `/interact/designlora` and `/finetune` share the same training + EI-sampling core;
> `/finetune` adds multi-checkpoint management on top.

## Repository layout

```
designlora_server.py       FastAPI service (the DesignLoRA loop)
dual_process/              Method core, adapted from g-luo/dual_process
  ├─ llm_surrogate.py        Bradley–Terry preference model  [DesignLoRA]
  ├─ dig_pipeline.py         edit construction + inner training loop
  ├─ dig_operators.py        VLM prompts + VQA losses
  ├─ dig_helpers.py          model loading + per-backbone helpers (incl. FLUX.2/Qwen)
  ├─ dig_viz.py, gpt_helpers.py   upstream, unchanged
  └─ UPSTREAM_CHANGES.md     provenance + exact change list
configs/                   base / pipe(generator) / vlm / app configs
image_utils.py, vlm_analysis_utils.py, vlm_query_visualizer.py, logging_utils.py
```

## License

The DesignLoRA code is released for research use. It builds on
[dual_process](https://github.com/g-luo/dual_process), which is subject to its own
license. Model weights are governed by their respective licenses — note in
particular that **FLUX.2 [klein] 9B is non-commercial**, while FLUX.2 [klein] 4B is
Apache-2.0. Review each model's license before use.

## Citing

```bibtex
@article{lee2026designlora,
  title   = {DesignLoRA: Interactive Fine-Tuning of Text-to-Image Models with Designer-in-the-Loop Feedback},
  author  = {Lee, Seung Won and Yun, Yejin and Hyun, Kyung Hoon},
  journal = {Advanced Engineering Informatics},
  year    = {2026}
}
```

DesignLoRA builds on Dual-Process Image Generation — please also cite:

```bibtex
@inproceedings{luo2025dualprocess,
  title     = {Dual-Process Image Generation},
  author    = {Luo, Grace and Granskog, Jonathan and Holynski, Aleksander and Darrell, Trevor},
  booktitle = {ICCV},
  year      = {2025}
}
```
