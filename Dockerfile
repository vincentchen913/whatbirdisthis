FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./server.py
COPY artifacts ./artifacts
COPY static ./static
COPY species_mapping.json ./species_mapping.json

ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
