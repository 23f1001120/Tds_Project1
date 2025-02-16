# Use slim version of Python 3.12 for smaller image
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# Download and install `uv`
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure `uv` is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Install Prettier globally (required for A2 task)
RUN npm install -g prettier@3.4.2

# Set working directory
WORKDIR /app

# Create data directory
RUN mkdir -p /data

# Set environment variable for dynamic path resolution
ENV DATA_DIR=/data

# Copy requirements.txt and install dependencies system-wide
COPY requirements.txt /app
RUN uv pip install -r /app/requirements.txt --system

# Copy the application code
COPY app.py /app

# Ensure app.py has execution permissions
RUN chmod +x /app/app.py

# Expose the API port (if running a FastAPI app)
EXPOSE 8000

# Run the application
CMD ["uv", "run", "app.py"]
