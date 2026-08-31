FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GARDEN_HOST=0.0.0.0 \
    GARDEN_DATA_DIR=/var/data

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /var/data

EXPOSE 8765
CMD ["python", "app.py", "--host", "0.0.0.0"]
