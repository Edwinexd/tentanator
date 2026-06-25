# syntax=docker/dockerfile:1
FROM python:3.13-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Talks to the backend over the compose network by default.
ENV TENTANATOR_API=http://backend:8787
CMD ["python", "app.py"]
