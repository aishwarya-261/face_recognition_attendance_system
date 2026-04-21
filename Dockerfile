FROM python:3.9-slim

# Install system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure some folders exist
RUN mkdir -p TrainingImage TrainingImageLabel StudentDetails Attendance

# Command to run the application
# We use Gradio's server setup in app.py
CMD ["python", "app.py"]
