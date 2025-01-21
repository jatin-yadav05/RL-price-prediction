# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements-deploy.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create directories for models and data
RUN mkdir -p models data

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "frontend/app.py"] 