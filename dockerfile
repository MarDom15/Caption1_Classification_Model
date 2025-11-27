# image Python légère
FROM python:3.11-slim

# répertoire de travail
WORKDIR /app

# Copier tout le projet dans l'image Docker
COPY . /app

# Mettre à jour pip et installer les dépendances
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r scripts/requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "scripts/apps/apps.py", "--server.port=8501", "--server.headless=true"]
