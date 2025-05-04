python -m pip install --upgrade pip
pip install --force-reinstall --ignore-installed --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.2.23
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT