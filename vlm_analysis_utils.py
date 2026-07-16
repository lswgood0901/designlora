import re
import json
import time
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image

from dual_process import dig_pipeline
from image_utils import add_score_to_image, _create_panel_with_winner
from vlm_query_visualizer import visualize_vlm_query


def analyze_selection_difference(sess: "Session", winner_idx: int) -> str:
    """선택된 이미지와 비선택 이미지들의 차이점을 VLM으로 분석.
    Returns a concise analysis of why the winner was selected."""
    if not sess.current_imgs or winner_idx >= len(sess.current_imgs):
        return "No analysis available."
    
    # Create a 2x2 grid showing all 4 images with winner marked
    panel_img = _create_panel_with_winner(sess.current_imgs, winner_idx)
    
    # Get BT scores for context (without descriptions)
    context_items = []
    for i, img in enumerate(sess.current_imgs):
        if i < len(sess.current_item_keys):
            key = sess.current_item_keys[i]
            score = sess.bt_scores.get(key, 0.5)
            context_items.append(f"Image {i+1}: preference {score*100:.1f}%")
    
    context_text = "\n".join(context_items)
    
    # Analysis prompt with full context
    analysis_prompt = (
        f"User selection history shows these 4 images with their preference probabilities:\n"
        f"{context_text}\n\n"
        f"The user selected Image {winner_idx + 1}. "
        f"Analyze the visual characteristics that make user preferred designs stand out. "
        f"List 3 specific visual qualities that explain why users favor this design. "
        f"Start with 'User preferred designs show:' and avoid Yes/No."
    )
    
    # System prompt for free-form analysis
    analysis_system = (
        "You are analyzing user image preferences. The user selected one image from four options. "
        "Identify specific visual characteristics that make the selected image stand out. "
        "Focus on observable differences in lighting, color, composition, texture, clarity, and style. "
        "Provide concrete, specific observations, not generic statements. "
        "DO NOT start with Yes or No. Start directly with the visual analysis."
    )
    
    # Query VLM for analysis
    try:
        # Use vlm_freeform_answer for descriptive analysis with more tokens
        analysis = vlm_freeform_answer(
            analysis_prompt, 
            panel_img,
            model_prompt=analysis_system,
            max_tokens=200,  # Increased for complete analysis
            sess=sess
        )
        
        # Save the analysis query image
        if sess:
            save_vlm_query_image(
                sess, panel_img, "selection_analysis", 
                f"Selection analysis for iter {sess.iter} - User selected Image {winner_idx + 1}"
            )
            # Save detailed VLM query visualization
            visualize_vlm_query(
                sess=sess,
                query_type="selection_analysis",
                question=analysis_prompt,
                answer="Analysis",
                main_image=panel_img,
                ref_images=[],
                metadata={
                    "iter": sess.iter,
                    "winner_idx": winner_idx,
                    "context_items": len(context_items)
                },
                model_prompt=analysis_system
            )
        
        # Validate and clean the response - remove Yes/No prefix if present
        if analysis:
            # Remove Yes/No prefix if it exists
            analysis_clean = analysis.strip()
            if analysis_clean.lower().startswith(('yes.', 'no.')):
                analysis_clean = analysis_clean[4:].strip()  # Remove "Yes." or "No."
            elif analysis_clean.lower().startswith(('yes,', 'no,')):
                analysis_clean = analysis_clean[4:].strip()  # Remove "Yes," or "No,"
            
            if len(analysis_clean.split()) > 5:
                return analysis_clean
        
        # If response is too short, try with more structured prompt
        retry_prompt = (
            f"The user preferred Image {winner_idx + 1} over the other three images. "
            f"Identify THREE specific visual qualities that characterize user preferred designs. "
            f"Format: 'User preferred designs feature: 1) [specific quality], 2) [specific quality], 3) [specific quality]'. "
            f"Do not start with Yes or No."
        )
        analysis = vlm_freeform_answer(retry_prompt, panel_img, model_prompt=analysis_system, max_tokens=200, sess=sess)
        
        if analysis:
            # Clean up Yes/No prefix from retry response too
            analysis_clean = analysis.strip()
            if analysis_clean.lower().startswith(('yes.', 'no.')):
                analysis_clean = analysis_clean[4:].strip()
            elif analysis_clean.lower().startswith(('yes,', 'no,')):
                analysis_clean = analysis_clean[4:].strip()
            
            if len(analysis_clean.split()) > 3:
                return analysis_clean
            
    except Exception as e:
        if sess.logger:
            sess.logger.warning(f"Selection analysis failed: {e}")
    
    return f"User preferred Image {winner_idx + 1} for its superior visual quality and composition."


def analyze_selection_history(sess: "Session") -> dict:
    """Analyze accumulated selection patterns from history.
    Returns structured insights about user preferences."""
    insights = {
        "preferred": [],
        "avoided": [],
        "consistency": "emerging",
        "selection_count": len(sess.history)
    }
    
    if not sess.history:
        return insights
    
    # Extract patterns from selection insights if available
    if hasattr(sess, 'selection_insights'):
        recent_insights = sess.selection_insights[-5:] if len(sess.selection_insights) > 5 else sess.selection_insights
        
        # Parse VLM analyses for common patterns
        all_features = []
        for insight in recent_insights:
            if 'analysis' in insight:
                # Extract features mentioned in analysis
                analysis_text = insight['analysis'].lower()
                # Common visual feature keywords
                feature_keywords = [
                    'lighting', 'color', 'texture', 'composition', 'contrast',
                    'brightness', 'saturation', 'sharpness', 'depth', 'perspective',
                    'warm', 'cool', 'soft', 'hard', 'natural', 'modern', 'minimal'
                ]
                for keyword in feature_keywords:
                    if keyword in analysis_text:
                        all_features.append(keyword)
        
        # Count feature frequencies
        if all_features:
            feature_counts = {}
            for f in all_features:
                feature_counts[f] = feature_counts.get(f, 0) + 1
            
            # Top preferred features (mentioned frequently)
            sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
            insights["preferred"] = [f for f, count in sorted_features[:4] if count >= 2]
    
    # Analyze BT score patterns for consistency
    if hasattr(sess, 'bt_scores') and len(sess.bt_scores) > 5:
        scores = list(sess.bt_scores.values())
        score_variance = np.var(scores) if scores else 0
        
        if score_variance < 0.05:
            insights["consistency"] = "highly consistent"
        elif score_variance < 0.1:
            insights["consistency"] = "moderately consistent"
        else:
            insights["consistency"] = "exploring preferences"
    
    return insights


def _extract_keywords(prompt: str) -> list[str]:
    """Extract visual keywords from a prompt for preference tracking."""
    # Remove common articles/prepositions and extract meaningful visual terms
    words = re.findall(r'\b\w+\b', prompt.lower())
    stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'that', 'this', 'these', 'those'}
    return [w for w in words if w not in stopwords and len(w) > 2]


def _build_preference_context(sess: "Session") -> tuple[str, str]:
    """Build natural language and JSON preference context from selection history."""
    if not sess.history:
        return "", ""
    
    # Extract keywords from recent selections (last 5 max)
    recent_selections = sess.history[-5:] if len(sess.history) > 5 else sess.history
    all_keywords = []
    for selection in recent_selections:
        if isinstance(selection, dict) and "prompt" in selection:
            all_keywords.extend(_extract_keywords(selection["prompt"]))
    
    if not all_keywords:
        return "", ""
    
    # Count frequency and get top preferences
    keyword_counts = {}
    for kw in all_keywords:
        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
    
    # Sort by frequency, take top items
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: -x[1])
    top_prefs = [kw for kw, count in sorted_keywords[:6] if count > 1]  # At least appeared twice
    
    if not top_prefs:
        # Fallback to most recent selection keywords
        top_prefs = list(set(all_keywords))[:4]
    
    # Build natural language summary
    if top_prefs:
        pref_text = f"User tends to prefer {', '.join(top_prefs[:3])}"
        if len(top_prefs) > 3:
            pref_text += f"; and elements like {', '.join(top_prefs[3:])}"
        pref_text += "."
    else:
        pref_text = "User preferences are still being learned."
    
    # Build JSON format
    pref_json = json.dumps({"prefer": top_prefs[:4], "avoid": []})
    
    return pref_text, pref_json


def _update_preference_profile(sess: "Session", selected_prompt: str, selected_img: Image.Image):
    """Update session's preference profile based on user selection."""
    # Add to history
    sess.history.append({"prompt": selected_prompt, "timestamp": time.time()})
    
    # Keep reference image (limit to last 3 to avoid memory bloat)
    sess.pref_images.append(selected_img)
    if len(sess.pref_images) > 3:
        sess.pref_images.pop(0)
    
    # Rebuild preference profile
    pref_text, pref_json = _build_preference_context(sess)
    sess.pref_profile = f"{pref_text}\nPreferences (JSON): {pref_json}"


def build_history_context(sess: "Session", max_refs: int = 4) -> tuple[str, dict, list[Image.Image]]:
    """Return (natural_summary, kv_dict, ref_images[<=max_refs]) from session history.
    Falls back gracefully when history/JSON is empty.
    """
    try:
        pref_text, pref_json = _build_preference_context(sess)
    except Exception:
        pref_text, pref_json = "", ""
    kv: dict = {}
    if pref_json:
        try:
            kv = json.loads(pref_json)
        except Exception:
            kv = {}
    # Natural text fallback if needed
    if not pref_text:
        if kv.get("prefer") or kv.get("avoid"):
            pref_text = (
                f"User prefers {', '.join(kv.get('prefer', []) or ['—'])}; "
                f"and avoids {', '.join(kv.get('avoid', []) or ['—'])}."
            )
        else:
            pref_text = "No prior preferences recorded."
    refs = list(getattr(sess, "pref_images", []) or [])[-max_refs:]
    return pref_text, kv, refs


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _looks_like_yesno(s: str) -> bool:
    t = _norm_text(s).lower().strip(".!? ").strip()
    return t in {"yes", "no"}


def _sanitize_caption(ans: str) -> str:
    cap = _norm_text(ans)
    if cap and cap[-1] not in ".!?":
        cap += "."
    return cap


def vlm_freeform_answer(
    question: str,
    img: Image.Image,
    model_prompt: str = "You are an image-captioning assistant. Respond with one complete English sentence based only on the image. Do not echo or paraphrase any provided text prompt.",
    max_tokens: int = 150,
    sess: "Session" = None
) -> str:
    # Need access to EDIT_BASE from the main module
    # This will be passed from the calling function
    if not hasattr(vlm_freeform_answer, 'edit_base'):
        return ""
    
    qa = [{"question": question, "answer": ""}]
    ed = dig_pipeline.create_edit(
        vlm_freeform_answer.pipe, 
        vlm_freeform_answer.edit_base["vlm"], 
        vlm_freeform_answer.edit_base["vlm_processor"], 
        vlm_freeform_answer.edit_base["cfg"], 
        qa, 
        model_prompt
    )
    try:
        ans = dig_pipeline.run_vlm(vlm_freeform_answer.pipe, ed, pred_x0=img, max_new_tokens=max_tokens)
    except Exception:
        ans = ""
    return (ans or "").strip()


def describe_image(img: Image.Image, prompt_text: str, sess: "Session" = None) -> str:
    CAP_SYS = (
        "You are an image-captioning assistant. Reply with one complete English sentence "
        "(8–20 words) describing only what is visible in the image. "
        "Do NOT answer Yes/No. Do NOT repeat or paraphrase the user's text prompt."
    )
    prompts = [
        f'Describe the image in one concise English sentence. Include key visual elements and the mood. Do not repeat or paraphrase this text: "{prompt_text}".',
        f'Provide a one-sentence English caption summarizing the scene\'s main objects, style, and lighting. Avoid using or paraphrasing this text: "{prompt_text}".',
        f'Write a short English caption (one sentence) describing the image content and atmosphere. Do not echo this text: "{prompt_text}".'
    ]
    # 1차 시도
    for q in prompts:
        ans = vlm_freeform_answer(q, img, CAP_SYS, max_tokens=80, sess=sess)  # Sufficient for caption
        if not ans:
            continue
        cap = _sanitize_caption(ans)
        if not _looks_like_yesno(cap) and len(cap.split()) >= 6:
            return cap
    # 2차 구제 시도(더 길게)
    retry_q = (
        f'Describe the image in one complete English sentence with at least 8 words. '
        f'Focus ONLY on what is visible. Do not repeat or paraphrase this text: "{prompt_text}".'
    )
    ans2 = vlm_freeform_answer(retry_q, img, CAP_SYS, max_tokens=80, sess=sess)
    if ans2:
        cap2 = _sanitize_caption(ans2)
        if not _looks_like_yesno(cap2) and len(cap2.split()) >= 6:
            return cap2
    # 폴백(아주 드묾)
    return _sanitize_caption(f"A depiction of {prompt_text.strip()}.")


# Helper function to set up VLM context (called from main module)
def setup_vlm_context(pipe, edit_base):
    """Setup the VLM context for the analysis functions."""
    vlm_freeform_answer.pipe = pipe
    vlm_freeform_answer.edit_base = edit_base


# Need to import save_vlm_query_image function
def save_vlm_query_image(sess, image, query_type, description="", include_score=False, score=None):
    """Placeholder - will be imported from main module to avoid circular import"""
    # This function will be replaced by the actual implementation from app_bt
    pass