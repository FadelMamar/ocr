# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

# Prepare working directory
WORKDIR /build

RUN apt-get update \
 && apt-get install -y poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements for caching
COPY requirements.txt ./

# # Install Python dependencies into /install
# ENV VIRTUAL_ENV=/opt/venv
# RUN uv venv /opt/venv
RUN uv pip install --system --no-cache-dir -r requirements.txt

# # ---------- Stage 2: Runtime ----------
# FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="fadel.seydou@gmail.com"
LABEL version="0.1.0"
LABEL description="OCR web service"

# Create workspace directory
WORKDIR /workspace

# Copy installed dependencies from builder
# COPY --from=builder /bin/uv /bin/uvx /bin/
# COPY --from=builder /opt/venv /opt/venv

# ENV VIRTUAL_ENV=/opt/venv
# ENV GOOGLE_API_KEY=
ENV MODEL="gemini/gemini-2.5-flash-preview-05-20"
ENV EXTRACTOR="dspy"
ENV TEMPERATURE=0.1

# Copy source code only
COPY src/ ./

# Expose ports
EXPOSE 4242
EXPOSE 8500

# Entrypoint
CMD ["./start_services.sh"]