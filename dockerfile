FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies except llama-cpp-python
COPY requirements.txt .
RUN pip install --upgrade pip

# If llama-cpp-python is in requirements.txt, REMOVE it there and install it manually
# Clone llama-cpp-python from source
RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git

# Install llama-cpp-python from source
WORKDIR /app/llama-cpp-python
RUN pip install .

# Return to app directory
WORKDIR /app

# Copy the rest of your code
COPY . .

# Install the remaining dependencies (excluding llama-cpp-python)
RUN pip install -r requirements.txt

# Expose your app port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
