FROM python:3.12.5-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Some networks block plain HTTP (port 80) or are flaky; prefer HTTPS mirrors.
RUN sed -i \
    -e 's|http://deb.debian.org|https://deb.debian.org|g' \
    -e 's|http://security.debian.org|https://security.debian.org|g' \
    /etc/apt/sources.list

# ---- System dependencies (geo stack + build tooling) ----
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
  && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
  && rm -rf /var/lib/apt/lists/*

# ---- Node.js (for React) ----
# Uses NodeSource for a recent Node LTS on Debian bullseye.
ARG NODE_MAJOR=20
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
  && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# ---- Python tooling ----
RUN python -m pip install --no-cache-dir -U pip

# Optional but recommended: centralize Python deps in requirements.txt
# Keep your current explicit pins here if you prefer, but requirements.txt scales better.

# ---- Python dependencies (from backend Dockerfile) ----
RUN pip install numpy==2.1.2
RUN pip install pandas==2.2.3
RUN pip install matplotlib==3.10.3
RUN pip install scipy==1.15.3
RUN pip install bokeh==3.7.3
RUN pip install argopy==1.2.0
RUN pip install uvicorn==0.34.3
RUN pip install fastapi==0.115.12
RUN pip install fsspec==2025.2.0
RUN pip install copernicusmarine==2.2.2
RUN pip install "xarray==2023.12.0" "zarr<3"
RUN pip install h5py==3.10.0

# Set working directory
WORKDIR /workspaces/app