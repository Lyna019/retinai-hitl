FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Borrow CUDA torch + torchvision + all NVIDIA libraries from the already-built
# model-service image. Docker copies from local image layers — no network needed.
COPY --from=retinai-hitl-model-service:latest \
    /usr/local/lib/python3.11/site-packages/ \
    /usr/local/lib/python3.11/site-packages/

# RetiZero source code (cloned on host, copied into image)
COPY RetiZero/ /app/RetiZero/

# Install only RetiZero-specific deps not already present in model-service
RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    pillow \
    "kornia==0.8.3" \
    timm

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache bert-base-cased tokenizer vocab so the container runs offline
RUN python -c "from transformers import BertTokenizerFast; BertTokenizerFast.from_pretrained('bert-base-cased')"

COPY taxonomy.yaml ./taxonomy.yaml
COPY app/ ./app/

EXPOSE 9001
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001"]
