# ==============================
# Base image
# ==============================
FROM python:3.10-slim

# ==============================
# Env settings
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# Set working directory
# ==============================
WORKDIR /app

# ==============================
# Install system dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Upgrade pip
# ==============================
RUN pip install --upgrade pip

# ==============================
# Install Python dependencies (CPU-only, tránh lỗi GPU)
# ==============================
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    xgboost==2.0.3 \
    lightgbm \
    matplotlib \
    seaborn \
    notebook

# ==============================
# Copy project
# ==============================
COPY . .

# ==============================
# Expose Jupyter port
# ==============================
EXPOSE 8888

# ==============================
# Default command (Jupyter)
# ==============================
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]