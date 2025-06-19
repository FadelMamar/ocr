class DoclingExtractor:
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
