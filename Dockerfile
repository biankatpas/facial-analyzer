# Imagem base Python otimizada
FROM python:3.10-slim

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY src/ /app/src/

# Criar diretórios para outputs
RUN mkdir -p /app/reports /app/models

# Baixar modelos do DeepFace antecipadamente
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Emotion')" || true

# Expor portas
EXPOSE 8000 8501

# Comando padrão
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]