# syntax=docker/dockerfile:1
FROM python:3.13-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user. The TUI is an interactive client with no host bind
# mount in docker-compose.yml (it only talks to the backend over the network),
# so it writes nothing to a host-owned path and dropping root is safe.
RUN adduser --disabled-password --gecos "" --uid 10001 app
USER app

# Talks to the backend over the compose network by default.
ENV TENTANATOR_API=http://backend:8787
CMD ["python", "app.py"]
