# Example script to run OCR using the Orchestrator from src/orchestrator.py
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import fire

from src.orchestrator import build_orchestrator

# Path to a sample image (adjust as needed)
SAMPLE_IMAGE_PATH = Path(__file__).parent.parent / "data/0-Immatriculation DGI NIF.jpg"


def load_sample_image():
    """Load the sample image for testing"""
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample image not found: {SAMPLE_IMAGE_PATH}"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()


def test_smoldocling(
    image_bytes=None,
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!",
):
    """Test SmolDoclingExtractor (default OCR extractor)"""
    print("\n--- Running SmolDoclingExtractor via Orchestrator ---")
    try:
        if image_bytes is None:
            image_bytes = load_sample_image()
        orchestrator_smoldocling = build_orchestrator(extractor_type="smoldocling")
        result_smoldocling = orchestrator_smoldocling.run(
            data=image_bytes, filetype="image", prompt=prompt
        )
        print(result_smoldocling)
        return result_smoldocling
    except Exception as e:
        print(f"SmolDoclingExtractor failed: {e}")
        return None


def test_rapidocr(
    image_bytes=None,
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!",
):
    """Test RapidOCRExtractor"""
    print("\n--- Running RapidOCRExtractor via Orchestrator ---")
    try:
        if image_bytes is None:
            image_bytes = load_sample_image()
        orchestrator_rapidocr = build_orchestrator(extractor_type="rapidocr")
        result_rapidocr = orchestrator_rapidocr.run(
            data=image_bytes, filetype="image", prompt=prompt
        )
        print(result_rapidocr)
        return result_rapidocr
    except Exception as e:
        print(f"RapidOCRExtractor failed: {e}")
        return None


def test_gemini(
    image_bytes=None,
    model="gemini-1.5-flash",
    temperature=0.7,
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!",
):
    """Test GeminiExtractor (requires model parameter)"""
    print("\n--- Running GeminiExtractor via Orchestrator ---")
    try:
        if image_bytes is None:
            image_bytes = load_sample_image()
        orchestrator_gemini = build_orchestrator(
            extractor_type="gemini", model=model, temperature=temperature
        )
        result_gemini = orchestrator_gemini.run(
            data=image_bytes, filetype="image", prompt=prompt
        )
        print(result_gemini)
        return result_gemini
    except Exception as e:
        print(f"GeminiExtractor failed: {e}")
        return None


def test_dspy(
    image_bytes=None,
    model="gemini/gemini-2.5-flash-preview-05-20",
    temperature=0.7,
    prompting_mode="cot",
    cache=True,
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!",
):
    """Test DspyExtractor (requires model parameter)"""
    print("\n--- Running DspyExtractor via Orchestrator ---")
    try:
        if image_bytes is None:
            image_bytes = load_sample_image()
        orchestrator_dspy = build_orchestrator(
            extractor_type="dspy",
            model=model,
            temperature=temperature,
            prompting_mode=prompting_mode,
            cache=cache,
        )
        result_dspy = orchestrator_dspy.run(
            data=image_bytes, filetype="image", prompt=prompt
        )
        print(result_dspy)
        return result_dspy
    except Exception as e:
        print(f"DspyExtractor failed: {e}")
        return None


def test_dolphin(
    image_bytes=None,
):
    """Test DolphinExtractor (ByteDance/Dolphin model)"""
    print("\n--- Running DolphinExtractor via Orchestrator ---")
    try:
        if image_bytes is None:
            image_bytes = load_sample_image()
        orchestrator_dolphin = build_orchestrator(extractor_type="dolphin")
        result_dolphin = orchestrator_dolphin.run(
            data=image_bytes, filetype="image", prompt=None
        )
        print(result_dolphin)
        return result_dolphin
    except Exception as e:
        print(f"DolphinExtractor failed: {e}")
        return None


def test_pdf(
    extractor_type="smoldocling",
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!",
):
    """Test PDF processing with specified extractor"""
    PDF_SAMPLE_PATH = Path(__file__).parent.parent / "data/sample.pdf"
    if not PDF_SAMPLE_PATH.exists():
        print(
            f"\n--- PDF processing skipped (no sample.pdf found at {PDF_SAMPLE_PATH}) ---"
        )
        return None

    print(f"\n--- Processing PDF with {extractor_type}Extractor ---")
    try:
        with open(PDF_SAMPLE_PATH, "rb") as f:
            pdf_bytes = f.read()

        # Build orchestrator based on extractor type
        if extractor_type == "gemini":
            orchestrator = build_orchestrator(
                extractor_type=extractor_type, model="gemini-1.5-flash", temperature=0.7
            )
        elif extractor_type == "dspy":
            orchestrator = build_orchestrator(
                extractor_type=extractor_type,
                model="gemini/gemini-2.5-flash-preview-05-20",
                temperature=0.7,
                prompting_mode="cot",
                cache=True,
            )
        else:
            orchestrator = build_orchestrator(extractor_type=extractor_type)

        result_pdf = orchestrator.run(data=pdf_bytes, filetype="pdf", prompt=prompt)
        print(f"PDF processing result (list of {len(result_pdf)} pages):")
        for i, page_result in enumerate(result_pdf):
            print(f"Page {i+1}: {page_result[:100]}...")
        return result_pdf
    except Exception as e:
        print(f"PDF processing failed: {e}")
        return None


def test_all(
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!"
):
    """Test all available extractors"""
    print("=== Testing All OCR Extractors ===")

    # Test all extractors
    results = {}
    results["smoldocling"] = test_smoldocling(prompt=prompt)
    results["rapidocr"] = test_rapidocr(prompt=prompt)
    results["gemini"] = test_gemini(prompt=prompt)
    results["dspy"] = test_dspy(prompt=prompt)
    results["dolphin"] = test_dolphin(prompt=prompt)

    # Test PDF processing
    results["pdf"] = test_pdf(prompt=prompt)

    print("\n=== Summary ===")
    for extractor, result in results.items():
        status = "✓ SUCCESS" if result is not None else "✗ FAILED"
        print(f"{extractor}: {status}")

    return results


def test_custom_image(
    image_path,
    extractor_type="smoldocling",
    prompt="Extract the text from the image. Reply directly. DO NOT TRANSLATE!",
    **kwargs,
):
    """Test a custom image with specified extractor"""
    print(
        f"\n--- Testing {extractor_type}Extractor with custom image: {image_path} ---"
    )

    # Load custom image
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return None

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Build orchestrator based on extractor type
        if extractor_type == "gemini":
            orchestrator = build_orchestrator(
                extractor_type=extractor_type,
                model=kwargs.get("model", "gemini-1.5-flash"),
                temperature=kwargs.get("temperature", 0.7),
            )
        elif extractor_type == "dspy":
            orchestrator = build_orchestrator(
                extractor_type=extractor_type,
                model=kwargs.get("model", "gemini/gemini-2.5-flash-preview-05-20"),
                temperature=kwargs.get("temperature", 0.7),
                prompting_mode=kwargs.get("prompting_mode", "cot"),
                cache=kwargs.get("cache", True),
            )
        else:
            orchestrator = build_orchestrator(extractor_type=extractor_type)

        result = orchestrator.run(data=image_bytes, filetype="image", prompt=prompt)
        print(result)
        return result
    except Exception as e:
        print(f"Custom image test failed: {e}")
        return None


def list_extractors():
    """List all available extractors"""
    extractors = {
        "smoldocling": "SmolDocling VLM pipeline via docling",
        "rapidocr": "RapidOCR pipeline via docling with custom models",
        "gemini": "Google Gemini Vision model (requires API key)",
        "dspy": "DSPy framework with various models (requires API key)",
        "dolphin": "ByteDance/Dolphin VLM model via HuggingFace",
    }

    print("Available OCR Extractors:")
    print("=" * 50)
    for name, description in extractors.items():
        print(f"{name:12} - {description}")
    print("=" * 50)


if __name__ == "__main__":
    fire.Fire(
        {
            "smoldocling": test_smoldocling,
            "rapidocr": test_rapidocr,
            "gemini": test_gemini,
            "dspy": test_dspy,
            "dolphin": test_dolphin,
            "pdf": test_pdf,
            "all": test_all,
            "custom": test_custom_image,
            "list": list_extractors,
        }
    )
