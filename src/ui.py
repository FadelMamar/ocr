# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:28:57 2025

@author: FADELCO
"""

import streamlit as st
from io import StringIO, BytesIO
from pdf2image import pdf2image
from PIL import Image
import requests
import base64
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OCR_SERVICE = "http://localhost:4242/predict"


@st.cache_data
def extract_data(
    data: bytes,
    prompt: str = None,
    out_images: list = None,
    url: str = "http://localhost:4242/predict",
):
    def send_one_image(image: Image.Image) -> str:
        # PIL -> bytes
        img_byte_arr = BytesIO()
        image.convert("RGB").save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()
        # bytes -> base64-encoding
        encoded_image = base64.b64encode(img_bytes).decode("utf-8")

        # payload
        payload = dict(image=encoded_image)
        if prompt:
            payload["prompt"] = prompt

        # send request
        response = requests.post(url=url, json=payload).json()

        return response.get("output")

    images = pdf2image.convert_from_bytes(data)

    if isinstance(out_images, list):
        out_images.extend(images)

    st.success(f"{len(images)} pages extracted from PDF.")

    with ThreadPoolExecutor(max_workers=2) as executor:
        out = list(executor.map(send_one_image, images))

    return out


# Streamlit UI
def main():
    st.title("Optical character recognition")

    # File upload widget
    with st.form("uploader"):
        uploaded_file = st.file_uploader("Upload a resume in PDF format", type=["pdf"])
        # prompt = st.text_input("Instruction",value="")

        if st.form_submit_button("Process"):
            extracted_images = []
            with st.spinner("Running...", show_time=True):
                # prompt = prompt if len(prompt)>1 else None
                outputs = extract_data(
                    data=uploaded_file.getvalue(),
                    prompt=None,
                    out_images=extracted_images,
                    url=OCR_SERVICE,
                )

            # st.write(outputs)

            # Create tabs for each page
            for i, image in enumerate(extracted_images):
                with st.expander(f"📄 Page {i + 1}", expanded=(i == 0)):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            image, caption=f"Page {i + 1}", use_container_width=True
                        )

                    with col2:
                        st.write(outputs[i])


if __name__ == "__main__":
    main()
