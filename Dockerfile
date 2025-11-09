# Define the base image as a build argument with a default value
ARG BASE_IMAGE=ubuntu:22.04
# Start from the specified base image
FROM ${BASE_IMAGE}

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Add build arguments for user and group IDs to match the host user
ARG UID=1000
ARG GID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Set the Blender download URL
ARG BLENDER_URL=https://mirror.clarkson.edu/blender/release/Blender4.5/blender-4.5.2-linux-x64.tar.xz

# Download and extract Blender
RUN wget -O blender.tar.xz ${BLENDER_URL} && \
    mkdir -p /opt/blender && \
    tar -xvf blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm blender.tar.xz

# Add Blender to the system's PATH for all users
ENV PATH="/opt/blender:${PATH}"

# Create a non-root user and grant passwordless sudo access
RUN groupadd -g ${GID} blender && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash blender && \
    usermod -aG sudo blender && \
    echo "blender ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/blender

# Switch to the non-root user
USER blender

# --- INSTALL PYTHON DEPENDENCIES ---
# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt /home/blender/requirements.txt

# Install python packages into the user's home directory
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r /home/blender/requirements.txt && rm /home/blender/requirements.txt

# Add the user's local bin to the PATH to make installed CLIs available
ENV PATH="/home/blender/.local/bin:${PATH}"

# Copy over an example script that uses blenderproc
COPY scripts/blenderproc_init.py /home/blender/blenderproc_init.py

# Run blenderproc on the example script to allow blenderproc to initialize itself
RUN blenderproc run /home/blender/blenderproc_init.py && rm /home/blender/blenderproc_init.py

# Copy over the blenderproc requirements file
COPY requirements_blenderproc.txt /home/blender/requirements_blenderproc.txt

# Install dependencies needed for `blenderproc run` commands
RUN blenderproc pip install $(cat /home/blender/requirements_blenderproc.txt) && rm /home/blender/requirements_blenderproc.txt

# Set the working directory
WORKDIR /home/blender/workspace