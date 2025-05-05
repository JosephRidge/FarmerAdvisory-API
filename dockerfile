FROM ubuntu:22.04  

RUN apt-get update && apt-get install -y \
    python3-pip \
    cmake \
    build-essential

WORKDIR /app
COPY . .

RUN pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu/ llama-cpp-python
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]