# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies required for geospatial libraries (GDAL, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your python script into the container
COPY main.py .

# Command to run when the container starts
ENTRYPOINT ["python", "main.py"]