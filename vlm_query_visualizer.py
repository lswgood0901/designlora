import os
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any


def visualize_vlm_query(
    sess: "Session",
    query_type: str,
    question: str,
    answer: str,
    main_image: Image.Image,
    ref_images: List[Image.Image] = None,
    metadata: Dict[str, Any] = None,
    model_prompt: str = ""
):
    """
    Save VLM query with detailed visualization showing:
    1. Main query image
    2. Reference images (if any)
    3. Question text
    4. Answer options
    5. Model prompt
    6. Metadata
    """
    if not sess or not sess.run_dir:
        return
    
    # Create VLM queries visualization directory
    vlm_viz_dir = Path(sess.run_dir) / "vlm_queries_detailed"
    vlm_viz_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    sess.vlm_query_counter += 1
    timestamp = time.strftime("%H%M%S")
    base_name = f"{sess.vlm_query_counter:04d}_{timestamp}_{query_type}"
    
    # Create a comprehensive visualization
    viz_image = create_query_visualization(
        question=question,
        answer=answer,
        main_image=main_image,
        ref_images=ref_images or [],
        model_prompt=model_prompt,
        metadata=metadata or {}
    )
    
    # Save visualization image
    viz_path = vlm_viz_dir / f"{base_name}_visualization.png"
    viz_image.save(viz_path)
    
    # Save main image separately
    main_img_path = vlm_viz_dir / f"{base_name}_main_image.png"
    main_image.save(main_img_path)
    
    # Save reference images separately
    ref_img_paths = []
    for i, ref_img in enumerate(ref_images or []):
        ref_path = vlm_viz_dir / f"{base_name}_ref_{i:02d}.png"
        ref_img.save(ref_path)
        ref_img_paths.append(str(ref_path))
    
    # Save query data as JSON
    query_data = {
        "timestamp": time.time(),
        "query_type": query_type,
        "question": question,
        "answer": answer,
        "model_prompt": model_prompt,
        "metadata": metadata or {},
        "files": {
            "visualization": str(viz_path),
            "main_image": str(main_img_path),
            "reference_images": ref_img_paths
        }
    }
    
    json_path = vlm_viz_dir / f"{base_name}_query_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(query_data, f, ensure_ascii=False, indent=2)
    
    print(f"📊 VLM Query Visualization saved: {base_name}")
    return str(viz_path)


def create_query_visualization(
    question: str,
    answer: str,
    main_image: Image.Image,
    ref_images: List[Image.Image],
    model_prompt: str,
    metadata: Dict[str, Any]
) -> Image.Image:
    """Create a comprehensive visualization of the VLM query."""
    
    # Configuration
    margin = 20
    text_height = 120
    ref_img_size = 150
    main_img_max_width = 400
    
    # Calculate main image size
    main_img = main_image.copy()
    if main_img.width > main_img_max_width:
        ratio = main_img_max_width / main_img.width
        new_height = int(main_img.height * ratio)
        main_img = main_img.resize((main_img_max_width, new_height), Image.LANCZOS)
    
    # Calculate canvas dimensions
    ref_images_width = len(ref_images) * (ref_img_size + margin) if ref_images else 0
    canvas_width = max(main_img.width + margin * 2, ref_images_width + margin * 2, 800)
    
    ref_images_height = ref_img_size + margin if ref_images else 0
    canvas_height = text_height * 3 + main_img.height + ref_images_height + margin * 6
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a better font
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    y_pos = margin
    
    # Title
    draw.text((margin, y_pos), "VLM Query Visualization", fill=(0, 0, 0), font=title_font)
    y_pos += 30
    
    # Metadata
    if metadata:
        meta_text = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
        draw.text((margin, y_pos), f"Metadata: {meta_text}", fill=(100, 100, 100), font=small_font)
        y_pos += 20
    
    # Model prompt section
    if model_prompt:
        draw.rectangle([margin-5, y_pos-5, canvas_width-margin+5, y_pos+text_height-5], 
                      fill=(255, 255, 200), outline=(200, 200, 0))
        draw.text((margin, y_pos), "Model Prompt:", fill=(0, 0, 0), font=title_font)
        y_pos += 25
        
        wrapped_prompt = wrap_text(model_prompt, canvas_width - margin * 2, text_font, draw)
        for line in wrapped_prompt:
            draw.text((margin, y_pos), line, fill=(50, 50, 50), font=text_font)
            y_pos += 15
        y_pos += margin
    
    # Question section
    draw.rectangle([margin-5, y_pos-5, canvas_width-margin+5, y_pos+text_height-5], 
                  fill=(200, 230, 255), outline=(0, 100, 200))
    draw.text((margin, y_pos), "Question:", fill=(0, 0, 0), font=title_font)
    y_pos += 25
    
    wrapped_question = wrap_text(question, canvas_width - margin * 2, text_font, draw)
    for line in wrapped_question:
        draw.text((margin, y_pos), line, fill=(0, 0, 0), font=text_font)
        y_pos += 15
    y_pos += margin
    
    # Answer section
    draw.rectangle([margin-5, y_pos-5, canvas_width-margin+5, y_pos+40], 
                  fill=(200, 255, 200), outline=(0, 150, 0))
    draw.text((margin, y_pos), f"Expected Answer: {answer}", fill=(0, 100, 0), font=title_font)
    y_pos += 50
    
    # Main image section
    draw.text((margin, y_pos), "Main Query Image:", fill=(0, 0, 0), font=title_font)
    y_pos += 25
    
    # Center the main image
    img_x = (canvas_width - main_img.width) // 2
    canvas.paste(main_img, (img_x, y_pos))
    y_pos += main_img.height + margin
    
    # Reference images section
    if ref_images:
        draw.text((margin, y_pos), f"Reference Images ({len(ref_images)}):", fill=(0, 0, 0), font=title_font)
        y_pos += 25
        
        # Arrange reference images in a row
        start_x = (canvas_width - len(ref_images) * ref_img_size - (len(ref_images) - 1) * margin) // 2
        for i, ref_img in enumerate(ref_images):
            # Resize reference image
            ref_resized = ref_img.copy()
            if ref_resized.width > ref_img_size or ref_resized.height > ref_img_size:
                ref_resized.thumbnail((ref_img_size, ref_img_size), Image.LANCZOS)
            
            # Center the image in the allocated space
            img_x = start_x + i * (ref_img_size + margin) + (ref_img_size - ref_resized.width) // 2
            img_y = y_pos + (ref_img_size - ref_resized.height) // 2
            
            # Draw border
            draw.rectangle([start_x + i * (ref_img_size + margin) - 2, y_pos - 2,
                           start_x + i * (ref_img_size + margin) + ref_img_size + 2, 
                           y_pos + ref_img_size + 2], outline=(150, 150, 150))
            
            canvas.paste(ref_resized, (img_x, img_y))
            
            # Add index label
            draw.text((start_x + i * (ref_img_size + margin) + 5, y_pos + 5), 
                     f"Ref {i+1}", fill=(255, 255, 255), font=small_font)
    
    return canvas


def wrap_text(text: str, max_width: int, font, draw) -> List[str]:
    """Wrap text to fit within the specified width."""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines