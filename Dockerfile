# Use an official Python lightweight image
FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8080 (the port magicpin might expect or you can map)
EXPOSE 8080

# Run the FastAPI server using Uvicorn on all network interfaces (0.0.0.0)
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8080"]
