import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap

# ===========================
#        Image Helpers
# ===========================
def display_text(width, height, text, scale=1.5, char_width=5, text_color="black", bg_color="white", font_path=None, font_size=20):
    old_width, old_height = width, height
    width = int(width / scale)
    height = int(height / scale)
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, font_size)
    header = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(header)
    
    char_width = 5
    max_chars_per_line = max(1, width // char_width)
    paragraphs = text.split('\n')
    text_lines = []
    for line in paragraphs:
        if not line.strip():
            text_lines.append('')
            continue
        wrapped = textwrap.fill(line, width=max_chars_per_line)
        text_lines.extend(wrapped.split('\n'))

    text_height_total = sum(draw.textbbox((0, 0), line, font=font)[3] for line in text_lines)
    text_y = (height - text_height_total) // 2
    for line in text_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, text_y), line, fill=text_color, font=font)
        text_y += text_bbox[3]
    return header.resize((old_width, old_height))

def stack_images(images):
    if not images: raise ValueError("Empty image list.")
    width, total_height = images[0].width, sum(img.height for img in images)
    if any(img.width != width for img in images): raise ValueError("Widths must match.")
    stacked_img = Image.new("RGB", (width, total_height))
    y_offset = 0
    for img in images:
        stacked_img.paste(img, (0, y_offset))
        y_offset += img.height
    return stacked_img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    images = [np.array(image) for image in images]
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img

def split_images(composite_image, num_rows, offset_ratio=0.02):
    composite_array = np.array(composite_image) if isinstance(composite_image, Image.Image) else composite_image
    h, w, _ = composite_array.shape
    row_height = h // num_rows - int(h * offset_ratio * (num_rows - 1)) // num_rows
    offset_h = int(h * offset_ratio)
    rows = []
    for i in range(num_rows):
        top = i * (row_height + offset_h)
        bottom = top + row_height
        row = composite_array[top:bottom, :]
        rows.append(Image.fromarray(row))
    return rows

def split_grid(collated, dim):
    images = []
    for row in split_images(collated, num_rows=dim[0]):
        col = split_images(row.transpose(Image.ROTATE_90), num_rows=dim[1])
        col = [item.transpose(Image.ROTATE_270) for item in col]
        images.append(col[::-1])
    return images

def resize_to_side(image, target_size, fn="min"):
    width, height = image.size
    if fn == "min":
        denom = min(width, height)
    elif fn == "max":
        denom = max(width, height)
    scale_factor = target_size / denom
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    return resized_image

# ===========================
#        Eval Helpers
# ===========================
def plot_losses(losses, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)
    image = Image.fromarray(image)
    return image