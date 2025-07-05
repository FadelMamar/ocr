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

## 📋 Table of Contents

- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Available Extractors](#-available-extractors)
- [Project Architecture](#-project-architecture)
- [Development](#-development)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)

---

## 🚀 Installation

### Prerequisites

* Python 3.11+
* Docker (optional)
* API keys for your chosen LLM provider

### Quickstart with uv (Recommended)

This project uses `uv` for dependency management. Install it first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/FadelMamar/ocr.git
cd ocr
uv sync
```

### Quickstart with pip

```bash
git clone https://github.com/FadelMamar/ocr.git
cd ocr
pip install -e .
```

### Quickstart with Docker

```bash
# Clone the repository
git clone https://github.com/FadelMamar/ocr.git
cd ocr

# Setup environment
cp example.env .env
# Edit .env with your API keys

# Run with Docker Compose
docker compose up
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `example.env`:

```bash
# Required API Keys (choose your provider)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=http://localhost:8000/v1

# Model Configuration
MODEL=gemini/gemini-2.5-flash-preview-05-20
EXTRACTOR=smoldocling
TEMPERATURE=0.7
```

### API Key Setup

#### Google Gemini
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Set `GOOGLE_API_KEY=your_key_here`

#### OpenAI/Claude
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an API key
3. Set `OPENAI_API_KEY=your_key_here`
4. Set `OPENAI_API_BASE` if using a custom endpoint

#### Local Models (Ollama)
1. Install [Ollama](https://ollama.ai/)
2. Pull your preferred model: `ollama pull llama3.2`
3. Use model names like `ollama_chat/llama3.2`

---

## 📡 API Documentation

### Endpoints

#### POST `/predict`

Extract text from images or PDFs using OCR.

**Request Body:**
```json
{
  "data": "base64_encoded_image_or_pdf",
  "prompt": "Extract the text from this image",
  "extractor": "smoldocling",
  "filetype": "image"
}
```

**Parameters:**
- `data` (required): Base64-encoded image or PDF bytes
- `prompt` (optional): Custom extraction prompt
- `extractor` (optional): OCR extractor type (default: `smoldocling`)
- `filetype` (optional): `"image"` or `"pdf"` (default: `"image"`)

**Response:**
```json
{
  "output": "Extracted text content"
}
```

### Usage Examples

#### Python Client

```python
import base64
import requests

# Encode image
with open("document.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    "http://localhost:4242/predict",
    json={
        "data": image_data,
        "prompt": "Extract all text from this document",
        "extractor": "gemini",
        "filetype": "image"
    }
)

print(response.json()["output"])
```

#### cURL Example

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 document.jpg)

# Make request
curl -X POST http://localhost:4242/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": "'$IMAGE_B64'",
    "prompt": "Extract all text from this document",
    "extractor": "smoldocling",
    "filetype": "image"
  }'
```

#### JavaScript/Node.js

```javascript
const fs = require('fs');

// Read and encode image
const imageBuffer = fs.readFileSync('document.jpg');
const imageBase64 = imageBuffer.toString('base64');

// Make request
fetch('http://localhost:4242/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    data: imageBase64,
    prompt: 'Extract all text from this document',
    extractor: 'smoldocling',
    filetype: 'image'
  })
})
.then(response => response.json())
.then(data => console.log(data.output));
```

---

## 🔧 Available Extractors

The service supports multiple OCR extractors, each optimized for different use cases:

### 1. SmolDoclingExtractor (Default)
- **Type**: VLM-based OCR
- **Best for**: High-quality text extraction with semantic understanding
- **Requirements**: None (works out of the box)
- **Performance**: Fast, good accuracy

### 2. RapidOCRExtractor
- **Type**: based on PaddleOCR
- **Best for**: Fast processing of standard documents
- **Requirements**: Downloads models on first use
- **Performance**: Very fast, moderate accuracy

### 3. GeminiExtractor
- **Type**: Google Gemini VLM
- **Best for**: Complex documents requiring interpretation
- **Requirements**: `GOOGLE_API_KEY`
- **Performance**: High accuracy, moderate speed

### 4. DspyExtractor
- **Type**: DSPy framework with multiple model support
- **Best for**: Advanced prompting and reasoning
- **Requirements**: Model configuration (Gemini, OpenAI, Ollama)
- **Performance**: High accuracy, flexible prompting


### Extractor Comparison

| Extractor | Speed | Accuracy | Setup | Best Use Case |
|-----------|-------|----------|-------|---------------|
| SmolDocling | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | General purpose |
| RapidOCR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Fast processing |
| Gemini | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Complex documents |
| Dspy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Advanced reasoning |
| Dolphin | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Structured documents |

---

## 🏗️ Project Architecture

### Codebase Structure

```
src/
├── app.py              # FastAPI/Litserve application
├── orchestrator.py     # Main orchestration logic
├── extractor.py        # OCR extractor implementations
├── loader.py           # Data loading utilities
└── ui.py              # Streamlit web interface

examples/
├── run_ocr.py         # CLI examples and testing
└── webservice.py      # Web service examples

data/                  # Sample images for testing
```

### Architecture Overview

The service follows a modular architecture:

1. **API Layer** (`app.py`): Handles HTTP requests and responses
2. **Orchestrator** (`orchestrator.py`): Coordinates between data loading and extraction
3. **Extractors** (`extractor.py`): Different OCR implementations
4. **Data Loader** (`loader.py`): Handles image/PDF loading and preprocessing
5. **UI** (`ui.py`): Streamlit web interface for easy testing

### Data Flow

```
Image/PDF → DataLoader → Orchestrator → Extractor → Response
```

---

## 🧪 Development

### Running the Service

#### Development Mode
```bash
# Start the API service
python src/app.py

# Start the Streamlit UI (in another terminal)
streamlit run src/ui.py
```

#### Using the Example Scripts

```bash
# Test all extractors
python examples/run_ocr.py test_all

# Test specific extractor
python examples/run_ocr.py test_smoldocling

# Test with custom image
python examples/run_ocr.py test_custom_image path/to/image.jpg
```

### Code Quality

The project uses `ruff` for linting and formatting:

```bash
# Check code quality
uvx ruff check src/

# Auto-fix issues
uvx ruff check --fix src/

# Format code
uvx ruff format src/
```

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Run benchmarks
python examples/run_ocr.py test_all
```

### Development Workflow

1. **Setup**: Clone repo and install dependencies with `uv sync`
2. **Configure**: Copy `example.env` to `.env` and add API keys
3. **Develop**: Use the modular architecture to add new extractors
4. **Test**: Use the example scripts to test functionality
5. **Format**: Run `uvx ruff check --fix src/` before committing

### Adding New Extractors

1. Create a new class in `src/extractor.py` inheriting from `Extractor`
2. Implement the `run(image: bytes, prompt: str) -> str` method
3. Add the extractor to `EXTRACTOR_MAP` in `orchestrator.py`
4. Update the factory function in `build_orchestrator()`
5. Add tests in `examples/run_ocr.py`

---

## 🧭 Roadmap

### ✅ Phase 1: Python MVP (In Progress)

* [x] Image upload endpoint (Litserve)
* [x] LLM integration to enhance or correct OCR output
* [x] Dockerized deployment
* [x] Multiple extractor support
* [x] Streamlit UI

### 🔜 Phase 2: Performance & Scalability

* [ ] Introduce Rust modules for performance hotspots (image decoding, pre/post-processing)
* [ ] Batch processing mode
* [ ] Async and queue-based inference
* [ ] CI/CD and monitoring integration

### 🔮 Phase 3: Business-Ready Features

* [ ] Multi-language OCR support
* [ ] Document structure detection (tables, forms)
* [ ] Advanced error handling and retry logic
* [ ] Performance monitoring and metrics

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ocr.git`
3. Install dependencies: `uv sync`
4. Create a feature branch: `git checkout -b feature/your-feature`
5. Make your changes and test with the example scripts
6. Format code: `uvx ruff check --fix src/`
7. Submit a pull request

### Code Style

- Follow PEP 8 with 88 character line length
- Use type hints for all function parameters and return values
- Add docstrings for all public functions and classes
- Run `uvx ruff check src/` before committing

### Testing

- Add tests for new extractors in `examples/run_ocr.py`
- Test with various image formats and quality levels
- Ensure error handling works correctly

---

## 📄 License

[Add your license information here]

---

## 🆘 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions for questions and ideas
- **Documentation**: Check the examples folder for usage patterns

---

## 🧠 Inspiration

This project draws on:

* The power of LLMs to understand context and correct OCR noise
* The need for enterprise-grade OCR tools that are fast, reliable, and easy to deploy

