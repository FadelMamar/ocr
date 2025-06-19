#!/bin/bash

# Start OCR service in the background
python app.py &

# Start Streamlit (UI)
streamlit run ui.py --server.port 8500

# Optional: wait to keep the script running until both services exit
wait