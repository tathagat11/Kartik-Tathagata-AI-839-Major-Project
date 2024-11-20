FROM python:3.9-slim

WORKDIR /app

# Install required packages
RUN pip install fastapi uvicorn mlflow transformers torch scikit-learn numpy

# Copy the server script
COPY server.py /app/

# Expose port
EXPOSE 5002

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5002"]