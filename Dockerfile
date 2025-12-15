FROM python:3.9-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. COPY EVERYTHING (Including the .joblib file)
COPY . .

# 3. Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]