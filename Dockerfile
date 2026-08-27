FROM python:3.11-slim

WORKDIR /app

# Install system libraries required by rasterio/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (allows Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml .
COPY data .
COPY Gather_GBIF_Sentinel_data.py .

CMD ["python", "Gather_GBIF_Sentinel_data.py"]