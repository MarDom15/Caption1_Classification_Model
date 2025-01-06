# Utiliser une image Python officielle comme base
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le contenu du répertoire 'scripts' dans le répertoire '/app' du conteneur
COPY scripts/ /app/

# Copier le répertoire 'Models' dans le conteneur sous '/app/Models'
COPY Models/ /app/Models/

# Installer les dépendances définies dans 'requirements.txt'
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exposer le port utilisé par Streamlit (par défaut 8501)
EXPOSE 8501

# Définir la commande pour exécuter l'application Streamlit depuis le dossier 'apps'
CMD ["streamlit", "run", "apps/apps.py"]
