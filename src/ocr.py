import dspy
import os
from abc import abstractmethod, ABC
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.llms.google_genai import GoogleGenAI


class ExtractorSignature(dspy.Signature):
    "Extract structured information from documents using Optical Character Recognition"

    pdf_image: dspy.Image = dspy.InputField(
        desc="Image with text as optical characters"
    )
    # prompt: str = dspy.InputField(desc="Instruction")
    extracted_text: str = dspy.OutputField(desc="Extracted text")


class Extractor(ABC):
    @abstractmethod
    def run(*args, **kwargs):
        pass


class GeminiExtractor(Extractor):
    def __init__(
        self, model: str = "gemini-2.5-flash-preview-04-17", temperature: float = 0.1
    ):
        self.llm = GoogleGenAI(model=model, temperature=temperature)

    def run(self, image: bytes, prompt: str = None):
        text = (
            prompt
            or "Extract the text from the image. Reply directly. DO NOT TRANSLATE!"
        )

        msg = ChatMessage(
            role="user",
            blocks=[
                ImageBlock(image=image),
                TextBlock(text=text),
            ],
        )

        resp = self.llm.chat([msg])

        text = resp.message.blocks[0].text

        return text


class DspyExtractor(Extractor):
    def __init__(
        self,
        model: str = "ollama_chat/gemma3:4b",
        temperature: float = 0.1,
        cache: bool = True,
        **kwargs,
    ):
        if model.startswith("ollama_chat"):
            lm = dspy.LM(
                model,
                api_key="",
                temperature=temperature,
                api_base="http://localhost:11434",
                cache=cache,
                **kwargs,
            )

        elif model.startswith("gemini"):
            api_key = os.environ.get("GOOGLE_API_KEY", None)
            if api_key is None:
                raise ValueError(
                    "Set environment variable 'GOOGLE_API_KEY' for Gemini model."
                )
            lm = dspy.LM(
                model, temperature=temperature, cache=cache, api_key=api_key, **kwargs
            )

        else:
            lm = dspy.LM(model, temperature=temperature, cache=cache, **kwargs)

        dspy.configure(lm=lm)
        self.llm = dspy.ChainOfThought(ExtractorSignature)

    def run(self, image: bytes, prompt: str = None):
        assert isinstance(image, bytes), (
            f"Expected type 'bytes' but found {type(image)}"
        )

        pdf_image = dspy.Image.from_file(image)

        response = self.llm(pdf_image=pdf_image)

        return response.extracted_text


class DoclingExtractor(Extractor):
    def __init__(
        self,
    ):
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, ImageFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        self.converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_cls=VlmPipeline,
                ),
            }
        )

    def run(self, image: bytes, prompt: str = None):
        doc = self.converter.convert(source=image).document

        return doc.export_to_markdown()


# if __name__ == "__main__":
#     from dotenv import load_dotenv

#     load_dotenv("../.env")

#     extractor = DspyExtractor(model='gemini/gemini-2.5-flash-preview-04-17',
#                               temperature=0.1,
#                               cache=False,
#                               )

#     with open("D:/workspace/repos/ocr/data/0-Immatriculation DGI NIF.jpg","rb") as image:
#         out = extractor.run(image=image.read())

#     print(out)
