# Paper ↔ code correspondence

How this repository maps to **DesignLoRA** (Lee et al., 2026), and where the
implementation differs from the paper. File:line references are to the cleaned tree.

## Correspondence

| Paper | Code |
|-------|------|
| §3.2 BTL preference history | `dual_process/llm_surrogate.py` (`bt_update`, `bt_score`, `bt_mle`); server `_recompute_bt_scores` (`designlora_server.py`) |
| §3.3 dual-prompt VLM eval (`p_question`, `p_pref`) | `dual_process/dig_operators.py:create_multi_image_vlm_prompt`; combined in `interact_designlora` / `finetune_lora` |
| §3.3 four references + BTL scores in prompt | `recent_eval_data[:4]` and `ref_bt_scores` in the server |
| §3.4 weighted VQA loss `L = -[w_q·log p_q + w_p·log p_p]`, `w_q=w_p=0.5` | `question_weight=pref_weight=0.5` (server); loss via `dig_pipeline.inner_loop` → `dig_operators.loss_vlm_multiqa` |
| §3.5 EI candidate sampling, `k=8`, `n=3` | `N_CAND = 8`, `N_EVAL = 3` (server) |
| §3.5 score/dispersion (Eq. 6), `ε_σ=0.01` | `sigma = np.std(yes_probs) + 0.01` (server) |
| §3.5 EI (Eq. 7) | `expected_improvement(mu, sigma, ...)` (server) |
| §4.2 20-step tuning per feedback | `N_CANDIDATES_PER_CYCLE = 20` (server) |
| §4.2 384×384 resolution | `height/width: 384` in `configs/pipe/*.yaml` |
| §4.2 FLUX.1-schnell + Idefics2 | `configs/pipe/schnell.yaml` + `configs/vlm/idefics2.yaml` (default) |
| Appendix A dual prompts | `create_multi_image_vlm_prompt` (preference) + question template |
| Appendix B FLUX.2 / Qwen-Image | `configs/pipe/flux2.yaml`, `configs/pipe/qwen_image.yaml` |

## Differences the author should be aware of

1. **EI reference point.** The paper (Eq. 7) sets `f_best` to *the highest preference
   score observed so far*. The code calls `expected_improvement(mu, sigma, best=0.5, ...)`
   with a **fixed** `best=0.5` (the neutral "Yes"-probability), so the exploitation term
   `(μ − f_best)` is measured against 0.5 rather than a running maximum. Candidate
   *ranking* within a round is unaffected (constant offset), but the exploration/
   exploitation balance differs from the paper. `UserSession.best_mu` exists but is not
   fed into the EI call.

2. **LoRA rank / learning rate defaults.** The paper's primary system (FLUX.1-schnell)
   uses **rank 8, α 8, lr 9e-5**. In the configs, `configs/base.yaml` sets `r: 16`,
   `lora_lr: 8e-6`, and `schnell.yaml` does **not** override them — so the effective
   schnell run is rank 16 / lr 8e-6, not the paper's stated values. (The FLUX.2 and
   Qwen-Image configs *do* set rank 16 / lr 9e-5, matching Appendix B.) Left unchanged so
   as not to silently alter your results; adjust `schnell.yaml` if you want the paper's
   main-study hyperparameters.

3. **α vs. `train_weight`.** The paper reports LoRA α = 8. The code scales LoRA via an
   `opt_kwargs.train_weight` multiplier (default 4–5 in `base.yaml` / the server) rather
   than a fixed α = 8; verify this matches your intended setup.

4. **Bradley–Terry gradient loss (removed).** Earlier code carried a separate BT logistic
   *gradient* loss (`one_step_bt_update` / `loss_bt_vlm`) that the live server never
   called. It is **not** the paper's objective (the paper trains with the dual VQA loss,
   Eq. 5) and was removed during cleanup. See `dual_process/UPSTREAM_CHANGES.md`.

5. **Only DesignLoRA is shipped.** The paper's User Study 1 compares three conditions —
   DesignLoRA, **Pref-Only** (selection only), and **Baseline** (question only). This
   release keeps only the full DesignLoRA system; the two ablation conditions
   (`prefonly`, `baseline`) were removed from the server.

6. **FLUX.2 [klein] is an extension.** The paper's Appendix B evaluates FLUX.2 [dev] and
   Qwen-Image. `flux2_klein` (the open-weight, step-distilled FLUX.2 [klein] 9B) is added
   here for a fully open-weight demo and was not part of the paper's experiments.

7. **Reference VLM in the panel prompt.** The paper describes the preference prompt as
   showing the four current-panel references with their BTL scores. The code uses the
   current panel's four images as references (`recent_eval_data[:4]`), consistent with
   the paper; note BTL scores accumulate globally across rounds while the VLM only ever
   sees the four most recent references (an intentional prompt-size choice, noted in the
   server comments).
