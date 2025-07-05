import logging
import os

import litserve as ls
from fastapi import HTTPException
from orchestrator import build_orchestrator

ls.configure_logging(use_rich=True)

logger = logging.getLogger("API")


class OCR(ls.LitAPI):
    def setup(self, device):
        # Default extractor type and model from environment
        self.default_extractor = os.environ.get("EXTRACTOR", "smoldocling")

    def decode_request(self, request):
        # Accepts: data (bytes or base64), prompt, extractor, filetype
        data = request.get("data", None)
        prompt = request.get("prompt", None)
        extractor = request.get("extractor", self.default_extractor)
        filetype = request.get("filetype", None)

        if data is None:
            raise HTTPException(status_code=500, detail="'data' is required in request")

        logger.info(f"Decoded request: extractor={extractor}, filetype={filetype}")

        return dict(data=data, prompt=prompt, extractor=extractor, filetype=filetype)

    def predict(self, x: dict):
        try:
            logger.info("Running OCR...")
            orchestrator = build_orchestrator(
                extractor_type=x["extractor"],
                temperature=float(os.environ.get("TEMPERATURE", 0.7)),
                model=os.environ.get("MODEL"),
                prompting_mode="basic",
                cache=True,
            )
            out = orchestrator.run(
                data=x["data"], filetype=x["filetype"], prompt=x["prompt"]
            )
            return out
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Internal error: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    def encode_response(self, output):
        # If output is an HTTPException, return error structure
        logger.info(f"Encoding response: {output}")
        return {"output": output}


if __name__ == "__main__":
    api = OCR()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=4242, pretty_logs=True, generate_client_file=False)
