import requests
import io
import base64
from PIL import Image
from typing import Union


def load_image(path_or_url: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image from a local path or a URL and return a PIL Image object.

    `path_or_url` is returned as is if it's an `Image` already.
    """
    if isinstance(path_or_url, Image.Image):
        return path_or_url
    elif path_or_url.startswith("http"):
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    return Image.open(path_or_url)

def convert_to_bytes(path_or_url: Union[str, Image.Image], b64_encode: bool = False) -> bytes:
    """Load an image from a path or URL and convert it to bytes."""
    image = load_image(path_or_url).convert("RGB")
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="PNG")
    image_bytes = image_bytes_io.getvalue()
    if not b64_encode:
        return image_bytes
    else:
        return base64.b64encode(image_bytes).decode("utf-8")