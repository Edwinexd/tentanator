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

# Run as a non-root user: this service shells out to pdflatex/pdftoppm on
# student-controlled content, so it is the most important one to drop root.
# uid 1000 matches the host owner of the bind-mounted data dir (./graded_exams),
# so the rendered results PDF can still be written; the expensive LaTeX/poppler
# work itself runs in tempfile.TemporaryDirectory() under /tmp (world-writable).
RUN adduser --disabled-password --gecos "" --uid 1000 app
USER app

ENV TENTANATOR_API=http://backend:8787 \
    TENTANATOR_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
