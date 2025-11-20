FROM python:3.13-slim

# Build argument to control installation method
# Values: "install" (from PyPI) or "compile" (from local source in editable mode)
ARG COMPILE_OR_INSTALL=compile

# Set working directory to allow the gui.py to be in the same directory as where we run
WORKDIR /app

# Install system dependencies required by wdoc
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID:GID 1000:1000
# This matches most default user IDs on Linux systems, allowing proper file permissions for volumes
RUN groupadd -g 1000 wdoc && \
    useradd -u 1000 -g 1000 -m -s /bin/bash wdoc

# Copy source code if we're doing a local install
# This copies the parent directory (the wdoc repository root)
COPY --chown=wdoc:wdoc ../ /tmp/wdoc-source/

# Install Python packages with conditional logic based on COMPILE_OR_INSTALL
# Using uv for faster installation, but pip is fine too
RUN pip install --no-cache-dir uv && \
    if [ "$COMPILE_OR_INSTALL" = "compile" ]; then \
        echo "Installing wdoc from local source in editable mode..." && \
        uv pip install --system -e /tmp/wdoc-source && \
        uv pip install --system gradio==5.49.1; \
    else \
        echo "Installing wdoc from PyPI..." && \
        uv pip install --system wdoc gradio==5.49.1; \
    fi && \
    rm -rf /root/.cache

# Copy the GUI script
COPY gui.py /app/gui.py

# Expose Gradio's default port
EXPOSE 7860

# Switch to non-root user for running the application
USER wdoc

# Create .local directory structure for the wdoc user and set permissions
RUN mkdir -p /home/wdoc/.local && \
    mkdir -p /home/wdoc/.cache/wdoc
USER root
RUN chown -R wdoc:wdoc /home/wdoc/.local && \
    chown -R wdoc:wdoc /home/wdoc/.cache/wdoc && \
    chown -R wdoc:wdoc /app
USER wdoc

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV HOME="/home/wdoc"

# Run the Gradio app
CMD ["python", "gui.py"]
