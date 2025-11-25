FROM python:3.12-slim

# Install system deps for wheel building
RUN apt-get update && apt-get install -y \
    gcc libc-dev libjpeg-dev libpng-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy & install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
