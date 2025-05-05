FROM python:3.9-slim

RUN apt-get update && apt-get install -y cmake build-essential

WORKDIR /app
COPY . .

RUN pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu/ llama-cpp-python
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]