FROM python:3.11-slim

# Set working directory to allow the gui.py to be in the same directory as where we run
WORKDIR /app

# Install system dependencies required by wdoc
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Using uv for faster installation, but pip is fine too
RUN pip install --no-cache-dir uv && \
    uv pip install --system wdoc gradio && \
    rm -rf /root/.cache

# Copy the GUI script
COPY gui.py /app/gui.py

# Expose Gradio's default port
EXPOSE 7860

# Set environment variable for Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the Gradio app
CMD ["python", "gui.py"]
