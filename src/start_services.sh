#!/bin/bash

# Start OCR service in the background
uv run python app.py &

# Start Streamlit (UI)
uv run streamlit run ui.py --server.port 8500

# Optional: wait to keep the script running until both services exit
wait