"""
Created on Wed Jun 18 16:28:57 2025

@author: FADELCO
"""

import base64

import requests
import streamlit as st

OCR_SERVICE = "http://localhost:4242/predict"


def extract_data(
    data: bytes,
    prompt: str,
    out_images: list,
) -> list[str]:
    """
    Extract data from image or PDF. If out_images is provided (as a list),
    it will be populated with images from the PDF (one per page).
    """
    from pdf2image import convert_from_bytes

    if isinstance(out_images, list):
        images = convert_from_bytes(data)
        out_images.clear()
        out_images.extend(images)

    encoded = base64.b64encode(data).decode("utf-8")
    payload = dict(data=encoded, prompt=prompt)
    response = requests.post(url=OCR_SERVICE, json=payload)
    try:
        response.raise_for_status()
        result = response.json()
        return result["output"]
    except Exception as e:
        st.error(f"Error from OCR service: {e}")
        return []


def main():
    st.set_page_config(page_title="OCR PDF Extractor", layout="wide")
    st.title("Optical Character Recognition (OCR) PDF Extractor")

    # Sidebar for file upload and prompt
    with st.sidebar:
        st.header("Upload PDF & Settings")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        process = st.button("Process PDF")

    if uploaded_file is not None and process:
        extracted_images = []
        with st.spinner("Extracting text and images from PDF..."):
            outputs = extract_data(
                data=uploaded_file.getvalue(),
                prompt=None,
                out_images=extracted_images,
            )
        if not outputs:
            st.warning("No output returned from OCR service.")
            return
        st.success(f"Extraction complete! {len(extracted_images)} page(s) processed.")
        # Create tabs for each page
        tabs = st.tabs([f"Page {i + 1}" for i in range(len(extracted_images))])
        for i, (tab, image) in enumerate(zip(tabs, extracted_images, strict=False)):
            with tab:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption=f"Page {i + 1}", use_container_width=True)
                with col2:
                    st.markdown(
                        f"**Extracted Text:**\n\n{outputs[i] if i < len(outputs) else ''}"
                    )
    else:
        st.info(
            "Please upload a PDF from the sidebar and click 'Process PDF' to begin."
        )


if __name__ == "__main__":
    main()
