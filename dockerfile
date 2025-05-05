FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Clone llama-cpp-python with submodules
RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
WORKDIR /app/llama-cpp-python
RUN pip install .

# Return to app directory
WORKDIR /app

# Copy app code
COPY . .

# Expose app port
EXPOSE 8000

# Run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
