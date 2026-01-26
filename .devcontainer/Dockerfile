FROM python:3.12.5-bullseye

# ---- System dependencies (geo stack + build tooling) ----
RUN apt-get update \
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
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# ---- Python tooling ----
RUN python -m pip install --no-cache-dir -U pip

# Optional but recommended: centralize Python deps in requirements.txt
# Keep your current explicit pins here if you prefer, but requirements.txt scales better.

# Core scientific stack
RUN pip install --no-cache-dir \
  numpy==2.1.2 \
  pandas==2.2.3 \
  scipy==1.15.3 \
  matplotlib==3.10.3 \
  bokeh==3.7.3

# xarray + storage backends
RUN pip install --no-cache-dir \
  "xarray==2023.12.0" \
  "zarr<3" \
  "fsspec==2025.2.0" \
  netCDF4

# Data sources
RUN pip install --no-cache-dir \
  argopy==1.2.0 \
  copernicusmarine==2.2.2

# Geo libs
RUN pip install --no-cache-dir \
  shapely \
  geopandas \
  cartopy

# API
RUN pip install --no-cache-dir \
  fastapi==0.115.12 \
  uvicorn==0.34.3