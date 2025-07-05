from .extractor import (
    DoclingExtractor,
    DolphinExtractor,
    DspyExtractor,
    GeminiExtractor,
    RapidOCRExtractor,
    SmolDoclingExtractor,
)
from .loader import DataLoader

EXTRACTOR_MAP = {
    "gemini": GeminiExtractor,
    "dspy": DspyExtractor,
    "smoldocling": SmolDoclingExtractor,
    "rapidocr": RapidOCRExtractor,
    "dolphin": DolphinExtractor,
}


class Orchestrator:
    def __init__(self, extractor_type: str = "smoldocling", **kwargs):
        self.loader = DataLoader()
        if isinstance(extractor_type, str):
            extractor_cls = EXTRACTOR_MAP.get(extractor_type.lower())
            if extractor_cls is None:
                raise ValueError(
                    f"Unknown extractor type: {extractor_type}. Available types: {EXTRACTOR_MAP.keys()}"
                )
            self.extractor = extractor_cls(**kwargs)
        else:
            raise ValueError("Extractor type must be a string")

    def run(self, data: bytes | str, filetype: str, prompt: str) -> str | list[str]:
        """
        Run extraction on the provided data using the configured extractor.
        Args:
            data: Input data (bytes or base64 string)
            filetype: 'pdf' or 'image'
            prompt: Extraction prompt
        Returns:
            Extracted text or list of texts (for PDFs)
        """
        # Load data (returns bytes for image, list of bytes for pdf)
        load_pdf_as_images = not isinstance(self.extractor, DoclingExtractor)
        loaded = self.loader.load(
            data, filetype=filetype, load_pdf_as_images=load_pdf_as_images
        )
        # If PDF, run extractor on each page
        if isinstance(loaded, list):
            return [self.extractor.run(image=img, prompt=prompt) for img in loaded]
        else:
            return self.extractor.run(image=loaded, prompt=prompt)


def build_orchestrator(
    model: str | None = None,
    extractor_type: str = "smoldocling",
    temperature: float = 0.7,
    prompting_mode: str = "basic",
    cache: bool = True,
) -> Orchestrator:
    """
    Factory function to build an Orchestrator with the correct extractor and arguments.
    Accepts extractor_type and passes relevant kwargs to the extractor.
    """
    extractor_type = extractor_type.lower()
    if extractor_type not in EXTRACTOR_MAP:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. Available: {list(EXTRACTOR_MAP.keys())}"
        )

    # Argument routing for each extractor
    extractor_args = {}
    if extractor_type == "gemini":
        assert isinstance(
            model, str
        ), f"Model is required for GeminiExtractor. Got {type(model)}"
        model = model.replace("gemini/", "")
        extractor_args["model"] = model
        extractor_args["temperature"] = temperature
    elif extractor_type == "dspy":
        assert isinstance(
            model, str
        ), f"Model is required for DspyExtractor. Got {type(model)}"
        assert isinstance(
            temperature, float
        ), f"Temperature must be a float. Got {type(temperature)}"
        assert prompting_mode in [
            "cot",
            "basic",
        ], f"Prompting mode must be 'cot' or 'basic'. Got {prompting_mode}"
        assert isinstance(cache, bool), f"Cache must be a boolean. Got {type(cache)}"
        extractor_args["model"] = model
        extractor_args["temperature"] = temperature
        extractor_args["prompting_mode"] = prompting_mode
        extractor_args["cache"] = cache
    # SmolDoclingExtractor and RapidOCRExtractor take no arguments

    return Orchestrator(extractor_type=extractor_type, **extractor_args)
