import base64
import logging
from pathlib import Path

import fire
import requests

logger = logging.getLogger("main")

EXAMPLE_FILE_PATH = "C:/Users/FADELCO/Downloads/FadelMSeydou_CV  (1).pdf"


def send(
    file_path: str = EXAMPLE_FILE_PATH,
    prompt: str = None,
    extractor="smoldocling",
    filetype="pdf",
):
    file_path = Path(file_path).resolve()

    logger.info(f"Parsing: {file_path}")

    with open(file_path, "rb") as file:
        encoded_file = base64.b64encode(file.read()).decode("utf-8")

    payload = dict(
        data=encoded_file, prompt=prompt, filetype=filetype, extractor=extractor
    )

    response = requests.post(url="http://localhost:4242/predict", json=payload)
    out = response.json()

    print(out["output"])


if __name__ == "__main__":
    fire.Fire({"extract": send})
