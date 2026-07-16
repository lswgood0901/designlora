import base64
from io import BytesIO

from dual_process import dig_viz

def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_message(prompt, image_key=None, images=[], image_size=224):
    if image_key is not None:
        parts = prompt.split(image_key)
    else:
        parts = [prompt]
    messages = [{"role": "user", "content": []}]
    if len(parts) == 1:
        messages[0]["content"].append({"type": "text", "text": parts[0]})
    elif len(parts) == len(images) + 1:
        for i, part in enumerate(parts):
            if part:
                messages[0]["content"].append({"type": "text", "text": part})
            if i < len(images):
                if type(images[i]) is str:
                    messages[0]["content"].append({
                        "type": "text",
                        "text": images[i]
                    })
                else:
                    image = images[i]
                    image = dig_viz.resize_to_side(image, image_size)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{pil_to_base64(image)}"}
                    })
    else:
        raise NotImplementedError
    return messages