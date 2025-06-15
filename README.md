# 🧠 OCR-as-a-Service (VLM-Powered Optical Character Recognition)

Welcome to the OCR project! This repository hosts a performant and extensible web service that performs **Optical Character Recognition (OCR)** using **Visual Language Models (VLMs)** via API calls. The initial implementation is in **Python**, with plans to introduce **Rust** for further performance gains.


---

## 🚀 Project Vision

Our goal is to create a **high-performance OCR web service** that:

* Extracts text from images using modern OCR techniques
* Leverages **VLMs** for enhanced interpretation and post-processing
* Prioritizes **speed, scalability, and robustness**
* Serves real-time and batch OCR use cases in business environments

---

## 🧾 What is OCR and Why It Matters

**Optical Character Recognition (OCR)** is the process of converting text from scanned documents, photos, PDFs, or image files into machine-readable text.

### 💼 Business Relevance

OCR is a **critical enabler of digital transformation**. It helps businesses:

* Automate data entry from paper forms or invoices
* Extract structured data from unstructured documents
* Enable search, indexing, and archiving of scanned files
* Improve accessibility and compliance

Industries like **finance, logistics, healthcare, law, and government** rely heavily on OCR to streamline operations and reduce manual processing time.

---

## 🧩 Problem This Project Solves

Despite existing solutions, many OCR tools:

* Struggle with **low-quality images**
* Lack **semantic understanding** of the extracted text
* Are hard to integrate or deploy as scalable web services
* Offer poor performance in real-time applications

This project addresses these limitations by:

* Using **VLMs** to interpret ambiguous or noisy text
* Designing a **modular web API** that's easy to extend
* Focusing on **low-latency and high throughput**
* Enabling multi-language and multi-format support

---

## 🛠️ Stack

| Layer             | Tooling                                            |
| ----------------- | -------------------------------------------------- |
| Language          | Python (Rust planned)                              |
| Model API         | OpenAI / Claude / Other LLM APIs                   |
| API Framework     | Litserve (Python)                                  |
| Performance Focus | Rust rewrite (planned) for speed-critical modules  |
| Testing           | Pytest + Benchmark tools                           |

---

## 🧭 Roadmap

### ✅ Phase 1: Python MVP (In Progress)

* [x] Image upload endpoint (Litserve)
* [x] LLM integration to enhance or correct OCR output
* [x] Dockerized deployment

### 🔜 Phase 2: Performance & Scalability

* [ ] Introduce Rust modules for performance hotspots (image decoding, pre/post-processing)
* [ ] Batch processing mode
* [ ] Async and queue-based inference
* [ ] CI/CD and monitoring integration

### 🔮 Phase 3: Business-Ready Features

* [ ] Multi-language OCR support
* [ ] Document structure detection (tables, forms)


## 📌 Getting Started

### Prerequisites

* Python 3.11+
* Docker (optional)
* OpenAI API key or other LLM provider key

### Quickstart (Python)

```bash
git clone https://github.com/FadelMamar/ocr.git
cd ocr
pip install -e .
python src/app.py
```

### Quickstart (Docker)

```bash
docker compose up
```
- make sure to rename ``example.env`` to ``.env`` and provide API keys.

---


## 🧠 Inspiration

This project draws on:

* The power of LLMs to understand context and correct OCR noise
* The need for enterprise-grade OCR tools that are fast, reliable, and easy to deploy

