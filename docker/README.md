# Docker Setup

This directory contains a dockerized Gradio web interface for wdoc, designed for easy deployment and use.

<p align="center"><img src="https://github.com/thiswillbeyourgithub/wdoc/blob/dev/images/gradio_interface.png?raw=true"></p>

## Prerequisites

This setup assumes you have already cloned the wdoc repository:
```bash
git clone https://github.com/thiswillbeyourgithub/wdoc.git
cd wdoc/docker
```

**Note**: No pre-built Docker images are provided. You'll build the image locally from the cloned repository.

## Quick Start

All commands below should be run from the `docker` subdirectory of the wdoc repository.

1. **Configure environment variables**: Copy and edit the environment file (both files are in the `./docker` directory):
   ```bash
   cp custom_env.example custom_env
   # Edit custom_env to add your API keys (OPENAI_API_KEY, etc.)
   ```

2. **Start the service**:
   ```bash
   docker-compose up -d
   ```

3. **Access the web interface**: Open your browser to `http://localhost:7618`

## Architecture

- **Build modes**: The Docker image can be built in two ways controlled by the `COMPILE_OR_INSTALL` build argument:
  - `compile` (default): Installs wdoc from the local repository source in editable mode. Use this for development or when you need the latest changes.
  - `install`: Installs wdoc from PyPI. Use this for a stable, released version.
  
  To change the build mode, set the environment variable before building:
  ```bash
  COMPILE_OR_INSTALL=install docker-compose up -d --build
  ```

- **Container user**: Runs as non-root user `wdoc` (UID:GID 1000:1000) for security
- **Port**: Exposes Gradio on port 7618 (mapped from internal port 7860)
- **Volumes** (relative to the `./docker` directory):
  - `./vectorstore`: Persistent storage for document embeddings
  - `./wdoc_cache`: LLM cache to reduce API costs and improve performance

## Troubleshooting

### Permission Errors

If you encounter permission errors on first startup, particularly related to the cache directory, this is typically because Docker created the volume directories with root ownership.

**Solution**: From the `docker` directory, change ownership to match the container's user (UID:GID 1000:1000):

```bash
# Make sure you're in the docker directory
cd wdoc/docker

# Fix permissions
sudo chown -R 1000:1000 ./vectorstore ./wdoc_cache

# Or if the directories don't exist yet:
mkdir -p ./vectorstore ./wdoc_cache
sudo chown -R 1000:1000 ./vectorstore ./wdoc_cache
```

**Alternative**: If you're running with a different user ID, you can modify the `docker-compose.yml` to use your current user:

```yaml
user: "${UID}:${GID}"
```

Then run with:
```bash
UID=$(id -u) GID=$(id -g) docker-compose up -d
```

### Checking Logs

To view the application logs:
```bash
docker-compose logs -f wdoc-gui
```

### Rebuilding After Changes

If you've modified `gui.py` or `Dockerfile`:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Configuration

### Environment Variables

Create a `custom_env` file in the `docker` directory with your configuration:

```bash
# Required: API keys for your LLM provider
OPENAI_API_KEY=sk-...
# Or for other providers:
# ANTHROPIC_API_KEY=...
# GEMINI_API_KEY=...

# Optional: Default models
WDOC_DEFAULT_MODEL=openai/gpt-4o-mini
WDOC_DEFAULT_EMBED_MODEL=openai/text-embedding-3-small

# Optional: Langfuse integration (if using)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Volume Paths

You can customize volume paths using environment variables in `docker-compose.yml`:

```bash
VECTORSTORE_PATH=/your/custom/path/vectorstore docker-compose up -d
CACHE_PATH=/your/custom/path/cache docker-compose up -d
```

## Security Notes

- The container runs as a non-root user for improved security
- Security option `no-new-privileges` prevents privilege escalation
- No unnecessary capabilities are granted
- Network access is controlled (uses `host.docker.internal` for local services like Langfuse)

## For Developers

### Building Locally

From the `docker` directory:

```bash
# Build from local source (default)
docker build -t wdoc-gui -f Dockerfile ..

# Or build from PyPI
docker build -t wdoc-gui -f Dockerfile --build-arg COMPILE_OR_INSTALL=install ..

# Run the container
docker run -p 7618:7860 \
  -v $(pwd)/vectorstore:/app/vectorstore \
  -v $(pwd)/wdoc_cache:/home/wdoc/.cache/wdoc \
  --env-file custom_env \
  wdoc-gui
```

### Modifying the GUI

The Gradio interface is defined in `docker/gui.py`. After making changes, rebuild the container to see them take effect.

### Understanding COMPILE_OR_INSTALL

- **`compile` mode**: The Dockerfile copies your local wdoc source code and installs it in editable mode (`pip install -e`). This means:
  - Code changes in the repository affect the Docker image after rebuild
  - Useful for development and testing
  - Includes unreleased features/fixes
  
- **`install` mode**: The Dockerfile installs wdoc from PyPI. This means:
  - You get the latest stable release
  - Independent of your local source code
  - Faster builds (no need to copy source files)

## Additional Resources

- [wdoc GitHub Repository](https://github.com/thiswillbeyourgithub/wdoc)
- [wdoc Documentation](https://wdoc.readthedocs.io/)
- [Gradio Documentation](https://gradio.app)

---

*This Docker setup was created with assistance from [aider.chat](https://github.com/Aider-AI/aider/)*
