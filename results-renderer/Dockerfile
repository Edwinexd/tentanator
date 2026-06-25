# syntax=docker/dockerfile:1
FROM python:3.13-slim

# LaTeX (answer sheets) + poppler (cover-page rasterisation for barcode decode).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      texlive-latex-base texlive-latex-recommended texlive-latex-extra \
      poppler-utils ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV TENTANATOR_API=http://backend:8787 \
    TENTANATOR_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
