# Define the base image as a build argument with a default value
ARG BASE_IMAGE=ubuntu:22.04
# Start from the specified base image
FROM ${BASE_IMAGE}

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libx11-6 \
    libxxf86vm1 \
    libxcursor1 \
    libxi6 \
    libxrandr2 \
    libxinerama1 \
    libegl1 \
    libwayland-client0 \
    libxkbcommon0 \
    wget \
    p7zip-full \
    unzip \
    tar \
    xz-utils \
    sudo \
    ca-certificates \
    libsm6 \
    libxext6 \
    libgl1 \
    python3 \
    python3-pip \
    git \
    libglib2.0-0 \
    libfontconfig1 \
    libxcb-icccm4 \
    libdbus-1-3 \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_BLENDER=false
# Set the Blender download URL
ARG BLENDER_URL=https://mirror.clarkson.edu/blender/release/Blender4.5/blender-4.5.2-linux-x64.tar.xz

# Download and extract Blender
RUN if [ "$INSTALL_BLENDER" = "true" ]; then \
    wget -O blender.tar.xz ${BLENDER_URL} && \
    mkdir -p /opt/blender && \
    tar -xvf blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm blender.tar.xz ; \
    else \
        echo "Skipping Blender installation." ; \
    fi

# Add Blender to the system's PATH for all users
ENV PATH="/opt/blender:${PATH}"

# Add build arguments for user and group IDs to match the host user
ARG UID=1000
ARG GID=1000

# Create a non-root user and grant passwordless sudo access
RUN groupadd -g ${GID} blender && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash blender && \
    usermod -aG sudo blender && \
    echo "blender ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/blender

# Switch to the non-root user
USER blender

# Set the working directory
WORKDIR /home/blender/workspace

# Add the user's local bin to the PATH to make installed CLIs available
ENV PATH="/home/blender/.local/bin:${PATH}"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ARG UV_SYNC_EXTRAS=cpu

# --- INSTALL PYTHON DEPENDENCIES ---
# Copy Python requirements and install them
COPY pyproject.toml uv.lock README.md .
RUN uv sync --frozen --no-dev --no-cache --extra ${UV_SYNC_EXTRAS} && rm pyproject.toml uv.lock README.md

# Copy over an example script that uses blenderproc
COPY scripts/blenderproc_init.py .

# Run blenderproc on the example script to allow blenderproc to initialize itself
RUN uv run blenderproc run blenderproc_init.py && rm blenderproc_init.py

# Copy over the blenderproc requirements file
COPY requirements_blenderproc.txt .

# Install dependencies needed for `blenderproc run` commands
RUN uv run blenderproc pip install $(cat requirements_blenderproc.txt) && rm requirements_blenderproc.txt
