# Étape 1 : Utiliser l'image Python 3.9-slim comme image de base
FROM python:3.9-slim


# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le contenu du répertoire 'scripts' dans le répertoire '/app' du conteneur
COPY scripts/ /app/scripts/

# Copier le répertoire 'Models' dans le conteneur sous '/app/Models'
COPY Models/ /app/Models/

# Installer les dépendances définies dans 'requirements.txt'
# On utilise 'scripts/requirements.txt' pour pointer correctement vers le fichier
RUN pip install --no-cache-dir -r /app/scripts/requirements.txt

# Exposer le port utilisé par Streamlit (par défaut 8501)
EXPOSE 8501

# Définir la commande pour exécuter l'application Streamlit depuis 'scripts/apps'
CMD ["streamlit", "run", "/app/scripts/apps/apps.py", "--server.port=8501", "--server.enableCORS=false"]

