from io import BytesIO
from PIL import Image
import numpy as np
import requests


def download_img(url: str) -> np.array:
    """
    Download image from url

    :param url: image url
    :return: image array RGB
    """
    data = requests.get(url)
    if data.status_code != 200:
        raise requests.HTTPError(f"Can't get image from url: {url}")
    response = requests.get(url)
    pil_img = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(pil_img)
