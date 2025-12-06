FROM python:3.11-slim    

WORKDIR /app

COPY requirements.txt .
COPY spam-seeker.py .
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "spam-seeker.py", "--camunda-worker"]
