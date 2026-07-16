import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _compose_side_by_side(left_img: Image.Image, right_img: Image.Image, target_h: int = 448) -> Image.Image:
    def _resize_keep_ar(img: Image.Image, h: int) -> Image.Image:
        w = int(img.width * (h / img.height))
        return img.resize((max(1, w), h), Image.BICUBIC)
    L = _resize_keep_ar(left_img, target_h)
    R = _resize_keep_ar(right_img, target_h)
    canvas = Image.new("RGB", (L.width + R.width, target_h), (255, 255, 255))
    canvas.paste(L, (0, 0)); canvas.paste(R, (L.width, 0))
    # 라벨(선택 사항)
    try:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.rectangle((4, 4, 70, 24), fill=(255,255,255))
        draw.text((8, 8), "Prev", fill=(0,0,0), font=font)
        draw.rectangle((L.width+4, 4, L.width+100, 24), fill=(255,255,255))
        draw.text((L.width+8, 8), "Candidate", fill=(0,0,0), font=font)
    except Exception:
        pass
    return canvas


def add_score_to_image(img: Image.Image, score: float, label: str = "preference probability", 
                      letter_label: str = None, is_winner: bool = None, vlm_confidence: float = None) -> Image.Image:
    """이미지에 preference probability (BT score) 텍스트, 라벨, winner/loser 상태를 추가.
    BT score는 사용자가 이 이미지를 선택할 상대적 확률을 나타냄 (0-1 범위)."""
    img_copy = img.copy()
    try:
        draw = ImageDraw.Draw(img_copy)
        # Try to use a better font if available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
            large_font = ImageFont.load_default()
        
        # Add winner/loser label at top if specified
        if is_winner is not None:
            status_text = "WINNER" if is_winner else "LOSER"
            status_color = (50, 205, 50) if is_winner else (220, 20, 60)  # Green for winner, red for loser
            
            bbox = draw.textbbox((0, 0), status_text, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img_copy.width - text_width) // 2
            y = 10
            
            # Draw background rectangle
            padding = 8
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=status_color
            )
            draw.text((x, y), status_text, fill=(255, 255, 255), font=large_font)
        
        # Add letter label (A, B, C, D) at top-left if specified
        if letter_label:
            bbox = draw.textbbox((0, 0), letter_label, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            padding = 6
            draw.rectangle(
                [10, 10, 10 + text_width + padding * 2, 10 + text_height + padding * 2],
                fill=(70, 130, 180)  # Steel blue
            )
            draw.text((10 + padding, 10 + padding), letter_label, fill=(255, 255, 255), font=large_font)
        
        # Add score text at bottom of image with percentage
        percentage = score * 100
        text = f"{label}: {percentage:.1f}%"
        
        # Add VLM confidence below if provided
        if vlm_confidence is not None:
            vlm_percentage = vlm_confidence * 100
            text += f"\nVLM confidence: {vlm_percentage:.1f}%"
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at bottom center
        x = (img_copy.width - text_width) // 2
        y = img_copy.height - text_height - 10
        
        # Draw background rectangle for better visibility
        padding = 5
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(255, 255, 255, 200)  # semi-transparent white
        )
        
        # Draw text
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
    except Exception as e:
        print(f"Failed to add score to image: {e}")
    
    return img_copy


def add_minimal_label_for_vlm(img: Image.Image, letter_label: str, is_winner: bool = None) -> Image.Image:
    """VLM 입력용: 문자 라벨과 WINNER/LOSER 상태만 추가 (preference probability 없이)"""
    img_copy = img.copy()
    try:
        draw = ImageDraw.Draw(img_copy)
        try:
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            large_font = ImageFont.load_default()
        
        # Add winner/loser label at top center if specified
        if is_winner is not None:
            status_text = "WINNER" if is_winner else "LOSER"
            status_color = (50, 205, 50) if is_winner else (220, 20, 60)  # Green for winner, red for loser
            
            bbox = draw.textbbox((0, 0), status_text, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img_copy.width - text_width) // 2
            y = 10
            
            # Draw background rectangle
            padding = 8
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=status_color
            )
            draw.text((x, y), status_text, fill=(255, 255, 255), font=large_font)
        
        # Add letter label (A, B, C, D) at top-left
        if letter_label:
            bbox = draw.textbbox((0, 0), letter_label, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            padding = 6
            draw.rectangle(
                [10, 10, 10 + text_width + padding * 2, 10 + text_height + padding * 2],
                fill=(70, 130, 180)  # Steel blue
            )
            draw.text((10 + padding, 10 + padding), letter_label, fill=(255, 255, 255), font=large_font)
    except Exception as e:
        print(f"Failed to add minimal label to image: {e}")
    
    return img_copy


def _vlm_augment_pil(img: Image.Image) -> Image.Image:
    """Light stochastic augmentations to induce variance in VLM scoring"""
    try:
        # 1) small random resize jitter around 360~420 px height
        h = random.randint(360, 420)
        w = int(img.width * (h / max(1, img.height)))
        img = img.resize((max(1, w), h), Image.BICUBIC)
        # 2) mild brightness/contrast jitter
        b = 0.95 + random.random() * 0.10  # 0.95~1.05
        c = 0.95 + random.random() * 0.10
        img = Image.eval(img, lambda px: int(min(255, max(0, ((px - 128) * c + 128) * b))))
        # 3) very light Gaussian-like noise
        if random.random() < 0.5:
            arr = np.array(img).astype(np.int16)
            noise = np.random.randint(-4, 5, size=arr.shape, dtype=np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
    except Exception:
        pass
    return img


def _create_panel_with_winner(images: list, winner_idx: int, target_size: int = 384) -> Image.Image:
    """Create 2x2 grid of images with winner marked."""
    # Resize all images to same size
    resized = []
    for img in images:
        h = target_size
        w = int(img.width * (h / img.height))
        resized.append(img.resize((w, h), Image.BICUBIC))
    
    # Create 2x2 canvas
    max_w = max(img.width for img in resized)
    canvas_w = max_w * 2 + 10  # 10px gap
    canvas_h = target_size * 2 + 10
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    
    # Place images in 2x2 grid
    positions = [(0, 0), (max_w + 10, 0), (0, target_size + 10), (max_w + 10, target_size + 10)]
    for i, (img, pos) in enumerate(zip(resized, positions)):
        canvas.paste(img, pos)
        
        # Mark the winner
        if i == winner_idx:
            try:
                draw = ImageDraw.Draw(canvas)
                # Draw "SELECTED" label on winner
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                label = "✓ SELECTED"
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = bbox[2] - bbox[0]
                label_h = bbox[3] - bbox[1]
                
                label_x = pos[0] + (img.width - label_w) // 2
                label_y = pos[1] + 10
                
                # Draw background
                padding = 5
                draw.rectangle(
                    [label_x - padding, label_y - padding, 
                     label_x + label_w + padding, label_y + label_h + padding],
                    fill=(50, 205, 50, 230)  # Green background
                )
                draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
                
                # Draw border around winner
                draw.rectangle(
                    [pos[0], pos[1], pos[0] + img.width - 1, pos[1] + img.height - 1],
                    outline=(50, 205, 50), width=3
                )
            except Exception:
                pass
        
        # Add index labels to all images
        try:
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
            idx_label = f"Image {i+1}"
            
            # Calculate text size to make box fit properly
            bbox = draw.textbbox((0, 0), idx_label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Add padding around text
            padding = 8
            box_width = text_width + padding * 2
            box_height = text_height + padding * 2
            
            # Draw background box with proper size
            draw.rectangle([pos[0] + 5, pos[1] + target_size - 5 - box_height, 
                           pos[0] + 5 + box_width, pos[1] + target_size - 5], 
                          fill=(255, 255, 255, 220))
            
            # Draw text centered in box
            text_x = pos[0] + 5 + padding
            text_y = pos[1] + target_size - 5 - box_height + padding
            draw.text((text_x, text_y), idx_label, fill=(0, 0, 0), font=font)
        except Exception:
            pass
    
    return canvas