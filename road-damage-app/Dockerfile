FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py engine.py ./
COPY web/ web/
COPY assets/ assets/

RUN mkdir -p weights \
    && curl -fL --retry 3 \
       https://github.com/Aaroh22/road-damage-detection/releases/download/model-v1.0/road_damage_yolo26s.pt \
       -o weights/road_damage_yolo26s.pt \
    && echo "2304871205864929cfbb86739fe0867021e56de456d33b30f99c0f1fa6ab8f27  weights/road_damage_yolo26s.pt" | sha256sum -c -

CMD ["sh", "-c", "exec python app.py --host 0.0.0.0 --port ${PORT:-8501}"]d