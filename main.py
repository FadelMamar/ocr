import fire
import requests
import base64
import json
import logging
from pathlib import Path

logger = logging.getLogger("main")

def send(image_file_path:str,prompt:str=None):

    image_file_path = Path(image_file_path).resolve()

    logger.info(f"Parsing: {image_file_path}")

    with open(image_file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    payload = dict(image=encoded_image)

    if prompt:
        payload["prompt"] = prompt

    response = requests.post(url="http://localhost:4242/predict",json=payload)
    out = json.loads(response.content)

    print(out["output"])



if __name__ == "__main__":
    fire.Fire()
