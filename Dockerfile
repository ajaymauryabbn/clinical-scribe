# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     ffmpeg     git     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements-backend.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user 	PATH=/home/user/.local/bin:$PATH

# Expose the port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
