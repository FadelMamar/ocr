import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO

import dspy
import numpy as np
import torch
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.io import DocumentStream
from huggingface_hub import snapshot_download
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.llms.google_genai import GoogleGenAI
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel


class ExtractorSignature(dspy.Signature):
    "Extract structured information from documents using Optical Character Recognition"

    pdf_image: dspy.Image = dspy.InputField(
        desc="Image with text as optical characters"
    )
    # prompt: str = dspy.InputField(desc="Instruction")
    extracted_text: str = dspy.OutputField(desc="Extracted text")


class Extractor(ABC):
    @abstractmethod
    def run(self, image: bytes, prompt: str = "") -> str:
        pass


class GeminiExtractor(Extractor):
    def __init__(
        self, model: str = "gemini-2.5-flash-preview-04-17", temperature: float = 0.1
    ):
        self.llm = GoogleGenAI(model=model, temperature=temperature)

    def run(self, image: bytes, prompt: str | None = None):
        text = (
            prompt
            if prompt is not None
            else "Extract the text from the image. Reply directly. DO NOT TRANSLATE!"
        )
        msg = ChatMessage(
            role="user",
            blocks=[
                ImageBlock(image=image),
                TextBlock(text=text),
            ],
        )
        resp = self.llm.chat([msg])
        # Find the first TextBlock in the response
        for block in resp.message.blocks:
            if isinstance(block, TextBlock):
                return block.text
        raise RuntimeError("No text block found in Gemini response")


class DspyExtractor(Extractor):
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        prompting_mode: str = "basic",
        cache: bool = True,
        **kwargs,
    ):
        model = model or os.environ.get("MODEL")

        assert (
            model is not None
        ), "Model is required. Set environment variable 'MODEL' or pass model as argument"

        lm = self.load_model(model, temperature, cache, **kwargs)

        dspy.configure(lm=lm)
        if prompting_mode == "cot":
            self.llm = dspy.ChainOfThought(ExtractorSignature)
        else:
            self.llm = dspy.Predict(ExtractorSignature)

    def load_model(
        self, model: str, temperature: float = 0.7, cache: bool = True, **kwargs
    ):
        if model.startswith("ollama_chat/"):
            lm = dspy.LM(
                model,
                api_key="",
                temperature=temperature,
                api_base="http://localhost:11434",
                cache=cache,
                **kwargs,
            )

        elif model.startswith("gemini/"):
            api_key = os.environ.get("GOOGLE_API_KEY", None)
            if api_key is None:
                raise ValueError(
                    "Set environment variable 'GOOGLE_API_KEY' for Gemini model."
                )
            lm = dspy.LM(
                model, temperature=temperature, cache=cache, api_key=api_key, **kwargs
            )

        elif model.startswith("openai/"):
            api_key = os.environ.get("OPENAI_API_KEY", None)
            api_base = os.environ.get("OPENAI_API_BASE", None)

            if api_key is None:
                raise ValueError(
                    "Set environment variable 'OPENAI_API_KEY' for OpenAI model."
                )
            if api_base is None:
                raise ValueError(
                    "Set environment variable 'OPENAI_API_BASE' for OpenAI model."
                )

            lm = dspy.LM(
                model,
                temperature=temperature,
                api_key=api_key,
                api_base=api_base,
                cache=cache,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown model: {model}. Supported models: ollama_chat/**, gemini/**, openai/**"
            )

        return lm

    def run(self, image: bytes, prompt: str | None = None):
        assert isinstance(
            image, bytes
        ), f"Expected type 'bytes' but found {type(image)}"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(image)
            tmp.flush()
            pdf_image = dspy.Image.from_file(tmp.name)
            response = self.llm(pdf_image=pdf_image)
        return response.extracted_text


class DoclingExtractor(Extractor):
    def __init__(self):
        self.converter = None

    def _preprocess(self, image: bytes):
        buf = BytesIO(image)
        source = DocumentStream(name="my_doc.pdf", stream=buf)
        return source

    def run(self, image: bytes, prompt: str = "") -> str:
        assert isinstance(
            self.converter, DocumentConverter
        ), f"Expected type 'DocumentConverter' but found {type(self.converter)}"
        source = self._preprocess(image)
        doc = self.converter.convert(source=source).document
        return doc.export_to_markdown()


class SmolDoclingExtractor(DoclingExtractor):
    """Extractor using SmolDocling VLM pipeline via docling."""

    def __init__(self):
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )


class RapidOCRExtractor(DoclingExtractor):
    """Extractor using RapidOCR pipeline via docling with custom models."""

    def __init__(self):
        # Download RapidOCR models from HuggingFace
        download_path = snapshot_download(repo_id="SWHL/RapidOCR")
        det_model_path = os.path.join(
            download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
        )
        rec_model_path = os.path.join(
            download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
        )
        cls_model_path = os.path.join(
            download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
        )
        ocr_options = RapidOcrOptions(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )
        pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_options,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )


class Dolphin:
    """https://github.com/bytedance/Dolphin/blob/master/demo_page_hf.py"""

    def __init__(self, model_id_or_path: str = "ByteDance/Dolphin"):
        """Initialize the Hugging Face model

        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()

        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()  # Always use half precision by default

        # set tokenizer
        self.tokenizer = self.processor.tokenizer

    def chat(self, prompt: str | list[str], image: Image.Image | list[Image.Image]):
        """Process an image or batch of images with the given prompt(s)

        Args:
            prompt: Text prompt or list of prompts to guide the model
            image: PIL Image or list of PIL Images to process

        Returns:
            Generated text or list of texts from the model
        """
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)

        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        # Prepare image
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.half().to(self.device)

        # Prepare prompt
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts, add_special_tokens=False, return_tensors="pt"
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)

        # Generate text
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
        )

        # Process output
        sequences = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )

        # Clean prompt text from output
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = (
                sequence.replace(prompts[i], "")
                .replace("<pad>", "")
                .replace("</s>", "")
                .strip()
            )
            results.append(cleaned)

        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results


@dataclass
class ImageDimensions:
    """Class to store image dimensions"""

    original_w: int
    original_h: int
    padded_w: int
    padded_h: int


class DolphinExtractor(Extractor):
    """Extractor using ByteDance/Dolphin VLM model via HuggingFace."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.model = Dolphin()
        self.save_dir = ".cache_dolphin"

    def _image_to_pil(self, image: bytes) -> Image.Image:
        return Image.open(BytesIO(image)).convert("RGB")

    def prepare_image(self, image: Image.Image) -> tuple[np.ndarray, ImageDimensions]:
        """Load and prepare image with padding while maintaining aspect ratio

        Args:
            image: PIL image

        Returns:
            tuple: (padded_image, image_dimensions)
        """
        import cv2

        try:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            original_h, original_w = image.shape[:2]

            # Calculate padding to make square image
            max_size = max(original_h, original_w)
            top = (max_size - original_h) // 2
            bottom = max_size - original_h - top
            left = (max_size - original_w) // 2
            right = max_size - original_w - left

            # Apply padding
            padded_image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            padded_h, padded_w = padded_image.shape[:2]

            dimensions = ImageDimensions(
                original_w=original_w,
                original_h=original_h,
                padded_w=padded_w,
                padded_h=padded_h,
            )

            return padded_image, dimensions
        except Exception as e:
            print(f"prepare_image error: {str(e)}")
            # Create a minimal valid image and dimensions
            h, w = image.height, image.width
            dimensions = ImageDimensions(
                original_w=w, original_h=h, padded_w=w, padded_h=h
            )
            # Return a black image of the same size
            return np.zeros((h, w, 3), dtype=np.uint8), dimensions

    def process_element_batch(self, elements, model, prompt, max_batch_size=None):
        """Process elements of the same type in batches"""
        results = []

        # Determine batch size
        batch_size = len(elements)
        if max_batch_size is not None and max_batch_size > 0:
            batch_size = min(batch_size, max_batch_size)

        # Process in batches
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i : i + batch_size]
            crops_list = [elem["crop"] for elem in batch_elements]

            # Use the same prompt for all elements in the batch
            prompts_list = [prompt] * len(crops_list)

            # Batch inference
            batch_results = model.chat(prompts_list, crops_list)

            # Add results
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                results.append(
                    {
                        "label": elem["label"],
                        "bbox": elem["bbox"],
                        "text": result.strip(),
                        "reading_order": elem["reading_order"],
                    }
                )

        return results

    def parse_layout_string(self, bbox_str):
        """Parse layout string using regular expressions"""
        pattern = (
            r"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)"
        )
        matches = re.finditer(pattern, bbox_str)

        parsed_results = []
        for match in matches:
            coords = [float(match.group(i)) for i in range(1, 5)]
            label = match.group(5).strip()
            parsed_results.append((coords, label))

        return parsed_results

    def process_elements(
        self,
        layout_results,
        padded_image,
        dims,
        model,
        max_batch_size,
        save_dir=None,
        image_name=None,
    ):
        """Parse all document elements with parallel decoding"""
        layout_results = self.parse_layout_string(layout_results)

        # Store text and table elements separately
        text_elements = []  # Text elements
        table_elements = []  # Table elements
        figure_results = []  # Image elements (no processing needed)
        previous_box = None
        reading_order = 0

        # Collect elements to process and group by type
        for bbox, label in layout_results:
            try:
                # Adjust coordinates
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    orig_x1,
                    orig_y1,
                    orig_x2,
                    orig_y2,
                    previous_box,
                ) = process_coordinates(bbox, padded_image, dims, previous_box)

                # Crop and parse element
                cropped = padded_image[y1:y2, x1:x2]
                if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                    if label == "fig":
                        pil_crop = Image.fromarray(
                            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        )

                        figure_filename = save_figure_to_local(
                            pil_crop, save_dir, image_name, reading_order
                        )

                        # For figure regions, store relative path instead of base64
                        figure_results.append(
                            {
                                "label": label,
                                "text": f"![Figure](figures/{figure_filename})",
                                "figure_path": f"figures/{figure_filename}",
                                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                                "reading_order": reading_order,
                            }
                        )
                    else:
                        # Prepare element for parsing
                        pil_crop = Image.fromarray(
                            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        )
                        element_info = {
                            "crop": pil_crop,
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }

                        # Group by type
                        if label == "tab":
                            table_elements.append(element_info)
                        else:  # Text elements
                            text_elements.append(element_info)

                reading_order += 1

            except Exception as e:
                print(f"Error processing bbox with label {label}: {str(e)}")
                continue

        # Initialize results list
        recognition_results = figure_results.copy()

        # Process text elements (in batches)
        if text_elements:
            text_results = self.process_element_batch(
                text_elements, model, "Read text in the image.", max_batch_size
            )
            recognition_results.extend(text_results)

        # Process table elements (in batches)
        if table_elements:
            table_results = self.process_element_batch(
                table_elements, model, "Parse the table in the image.", max_batch_size
            )
            recognition_results.extend(table_results)

        # Sort elements by reading order
        recognition_results.sort(key=lambda x: x.get("reading_order", 0))

        return recognition_results

    def process_single_image(
        self, image: bytes, save_dir, image_name, max_batch_size=None
    ):
        """Process a single image (either from file or converted from PDF page)

        Args:
            image: PIL Image object
            model: DOLPHIN model instance
            save_dir: Directory to save results
            image_name: Name for the output file
            max_batch_size: Maximum batch size for processing
            save_individual: Whether to save individual results (False for PDF pages)

        Returns:
            Tuple of (json_path, recognition_results)
        """
        # Stage 1: Page-level layout and reading order parsing
        pil_image = self._image_to_pil(image)
        layout_output = self.model.chat(
            "Parse the reading order of this document.", pil_image
        )

        # Stage 2: Element-level content parsing
        padded_image, dims = self.prepare_image(pil_image)
        recognition_results = self.process_elements(
            layout_output,
            padded_image,
            dims,
            self.model,
            max_batch_size,
            save_dir,
            image_name,
        )

        return recognition_results

    @torch.no_grad()
    def run(self, image: bytes, prompt: str | None = None) -> str:
        """
        Extract text from image using Dolphin model.

        Args:
            image: Image bytes
            prompt: Optional prompt for text extraction (not used by Dolphin)

        Returns:
            Extracted text from the image
        """
        assert isinstance(
            image, bytes
        ), f"Expected type 'bytes' but found {type(image)}"

        try:
            # Preprocess image
            results = self.process_single_image(
                image, save_dir=self.save_dir, image_name="image", max_batch_size=None
            )
            return results[0]
        except Exception as e:
            error_msg = f"Dolphin text extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
