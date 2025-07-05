import base64
from io import BytesIO

from pdf2image import convert_from_bytes
from PIL import Image


class DataLoader:
    """
    Handles loading of images (bytes or base64) and PDFs (bytes).
    For PDFs, converts each page to an image using pdf2image.
    """

    def load_image(self, data: bytes | str) -> bytes:
        """
        Load an image from bytes or base64 string. Returns image bytes (JPEG).
        """
        if isinstance(data, str):
            # Assume base64 encoded
            data = base64.b64decode(data)
        # Validate image by opening and re-saving as JPEG
        img = Image.open(BytesIO(data)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def load_pdf(
        self, data: bytes | str, load_pdf_as_images: bool = False
    ) -> bytes | list[bytes]:
        """
        Convert PDF bytes or base64-encoded PDF string to a list of image bytes (JPEG, one per page).
        """
        if isinstance(data, str):
            # Assume base64 encoded PDF
            data = base64.b64decode(data)

        assert isinstance(data, bytes), "Data must be bytes or base64-encoded string"

        if load_pdf_as_images:
            result = []
            images = convert_from_bytes(data)
            for img in images:
                buf = BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                result.append(buf.getvalue())
        else:
            result = data

        return result

    def guess_filetype(self, data: bytes | str) -> str:
        """
        Guess filetype from data.
        """
        if isinstance(data, str):
            data = base64.b64decode(data)
        if data[:4] == b"%PDF":
            return "pdf"
        else:
            return "image"

    def load(
        self, data: bytes | str, filetype: str = None, load_pdf_as_images: bool = False
    ) -> bytes | list[bytes]:
        """
        Generic loader. If filetype is 'pdf', treat as PDF. If 'image', treat as image.
        If filetype is None, tries to guess from content.
        """
        if filetype is None:
            filetype = self.guess_filetype(data)
        if filetype == "pdf":
            return self.load_pdf(data, load_pdf_as_images)
        elif filetype == "image":
            return self.load_image(data)
        else:
            raise ValueError(f"Unknown filetype: {filetype}")
