# On part de l'image de base
FROM jupyter/pyspark-notebook

# On passe en root pour installer les dépendances système si nécessaire
USER root

# (Optionnel) Utile pour certaines librairies géo comme geopandas
RUN apt-get update && apt-get install -y build-essential

# On repasse en utilisateur jovyan
USER ${NB_UID}

# --- CORRECTION ICI ---
# On va chercher le fichier dans le dossier 'work'
COPY --chown=${NB_UID}:${NB_GID} work/requirements.txt /tmp/

# Installation
RUN pip install --quiet --no-cache-dir -r /tmp/requirements.txt