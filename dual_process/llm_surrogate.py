"""
Clean LLM Surrogate for Image Preference Evaluation

Essential functions only:
• rank_images(prompt, images, top_k) → top images with scores
• bt_update(pairs) → update Bradley-Terry model  
• bt_score(item_id) → get BT probability
• attribute_edit(prompt) → generate prompt variations
"""

# [DesignLoRA] Rewritten by DesignLoRA into the Bradley-Terry preference surrogate
# (BTL goodness scores feed the preference-aware VLM prompt). See UPSTREAM_CHANGES.md.

import base64
import io
import asyncio
import openai
import math
import numpy as np
import random
import time
import os
from PIL import Image as PILImage
from scipy.special import softmax
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from openai import AsyncOpenAI


SURRO_MODEL = os.getenv("SURROGATE_MODEL", "gpt-4o-mini")

def set_surrogate_model(name: str):
    """Set surrogate model (e.g., 'gpt-4o-mini', 'gpt-4o')"""
    global SURRO_MODEL
    SURRO_MODEL = name

# Global Bradley-Terry state
all_pairs: List[Tuple[str, str]] = []  # [(winner, loser), ...]
all_prompts: set[str] = set()
_theta = {}  # image_id -> theta value


# Bradley-Terry Implementation
def bt_to_prob(theta: Dict[str, float], temp: float = 1.0) -> Dict[str, float]:
    """Convert theta (log-ability) to probabilities"""
    exp_vals = {p: math.exp(v / temp) for p, v in theta.items()}
    Z = sum(exp_vals.values()) + 1e-9
    return {p: v / Z for p, v in exp_vals.items()}


def bt_mle(pairs: List[Tuple[str, str]], items: List[str], lam: float = 0.05) -> Dict[str, float]:
    """Bradley-Terry maximum likelihood estimation"""
    idx = {item: i for i, item in enumerate(items)}
    
    def nll(theta):
        loss = 0.0
        for winner, loser in pairs:
            tw, tl = theta[idx[winner]], theta[idx[loser]]
            loss -= math.log(math.exp(tw) / (math.exp(tw) + math.exp(tl)))
        loss += (lam / 2.0) * np.sum(theta**2)  # L2 penalty
        return loss

    res = minimize(nll, np.zeros(len(items)), method="L-BFGS-B")
    return {item: float(res.x[i]) for item, i in idx.items()}


def bt_update(pairs: List[Tuple[str, str]]):
    """Add pairs to global BT state"""
    global all_pairs, all_prompts
    all_pairs.extend(pairs)
    for w, l in pairs:
        all_prompts.add(w)
        all_prompts.add(l)


def bt_score(item: str, temp: float = 1.0) -> float:
    """Get BT probability for item. Returns 0.5 if no data."""
    if item not in all_prompts:
        return 0.5
    if not all_pairs:
        return 1.0 / len(all_prompts)
    th = bt_mle(all_pairs, list(all_prompts))
    probs = bt_to_prob(th, temp)
    return probs.get(item, 0.5)


# Prompt Attribute Editing
def attribute_edit(
    prompt: str,
    exclude: List[str] = None,
    min_n: int = 1,
    max_n: int = 3,
    n_return: int = 5,
    max_retry: int = 3,
) -> List[str]:
    """Generate variations of prompt with 1-3 attributes changed"""
    ATTRS = ["style", "color", "material", "era", "size", "pattern"]
    exclude = set(exclude or [])
    
    sys_msg = (
        "You are a prompt rewriter for interior design.\n"
        f"Randomly change BETWEEN {min_n} AND {max_n} attributes from this list:\n"
        f"{ATTRS}\n"
        "Never touch attributes outside the list.\n"
        "Keep all other words identical.\n"
        "Return only the modified prompt."
    )
    user_msg = f"ORIGINAL:\n{prompt}\nNEW:"

    client = openai.OpenAI()
    for _ in range(max_retry):
        try:
            rsp = client.chat.completions.create(
                model=SURRO_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                n=n_return,
                temperature=0.3,
                top_p=0.9,
                max_tokens=64,
                seed=random.randint(1, 2**31 - 1),
            )

            outs = [
                ch.message.content.strip()
                for ch in rsp.choices
                if ch.message.content
                and ch.message.content.lower() != prompt.lower()
                and ch.message.content not in exclude
            ]
            if outs:
                return outs
        except Exception:
            time.sleep(0.8)
    return []


# Image Evaluation  
async def _llm_pref(
    prompt: str, 
    images: List[PILImage.Image], 
    bt_scores: Dict[str, float] = None
) -> np.ndarray:
    """Get preference probabilities for images using LLM"""
    def pil_to_datauri(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    # Build evaluation text with clear criteria
    eval_text = f"TARGET PROMPT: {prompt}\n\n"
    
    if bt_scores:
        eval_text += "BRADLEY-TERRY PREFERENCE SCORES:\n"
        eval_text += "These scores represent relative user preferences from past selections.\n"
        eval_text += "Higher scores indicate images users have historically preferred.\n"
        for i, (img_id, score) in enumerate(bt_scores.items()):
            eval_text += f"Image {i+1}: BT score = {score:.3f}\n"
        eval_text += "\n"
    
    eval_text += (
        "EVALUATION CRITERIA:\n"
        "1. Prompt alignment: How well does the image match the text description?\n"
        "2. Visual quality: Composition, lighting, clarity, aesthetics\n"
        "3. Technical execution: Proper proportions, realistic details\n"
        "4. Overall appeal: Would users find this image attractive/desirable?\n\n"
        "Rate each image based on these criteria. "
        "Return **only** a list of floating-point scores (0-1), one per image."
    )

    user_content = [{"type": "text", "text": eval_text}]
    for img in images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": pil_to_datauri(img), "detail": "low"}
        })

    try:
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=SURRO_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an expert image evaluator. Assess images based on "
                        "prompt alignment, visual quality, technical execution, and appeal. "
                        "Return only numerical scores as a list."
                    )
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=64,
            temperature=0.2,
        )
        scores = resp.choices[0].message.content.strip()
        scores = np.array(eval(scores))
    except Exception:
        scores = np.ones(len(images))
    
    probs = softmax(scores)
    return np.atleast_1d(probs).astype(float)


async def rank_images(
    prompt: str, 
    images: List[PILImage.Image], 
    top_k: int = 4,
    bt_scores: Dict[str, float] = None
) -> Tuple[List[PILImage.Image], np.ndarray]:
    """Rank images by preference and return top k"""
    probs = await _llm_pref(prompt, images, bt_scores)
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [images[i] for i in top_idx], probs[top_idx]


# Synchronous wrapper for rank_images (backwards compatibility)
def sync_rank_images(prompt: str, images: List[PILImage.Image], top_k: int = 4):
    """Synchronous version of rank_images"""
    return asyncio.run(rank_images(prompt, images, top_k))


# For backwards compatibility, keep these aliases
a_rank_images = rank_images


def bt_clear():
    """Clear all Bradley-Terry state - useful when switching between systems"""
    global all_pairs, all_prompts, _theta
    all_pairs.clear()
    all_prompts.clear()
    _theta.clear()
    print("🔄 BT state cleared: all_pairs, all_prompts, and _theta reset")