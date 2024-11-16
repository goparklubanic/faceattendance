# Use official Python image
FROM python:3.11.9-slim-bookworm

# Install dependencies including CMake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 libxrender-dev \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    && apt-get clean

# Install Python libraries
RUN pip install --upgrade pip
RUN pip install numpy opencv-python-headless
RUN pip install pybind11>=2.12

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose the port and start Flask app
EXPOSE 5000
CMD ["python", "app.py"]
