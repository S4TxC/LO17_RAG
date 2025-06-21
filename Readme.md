# LO17 - Projet N°2

L'objectif de ce second projet est de réaliser un système de Retrieval Augmented Generation (RAG) sur le sujet de notre choix. Nous avons axé notre projet sur l'analytique prédictive pour l'éducation. Notre RAG compile un ensemble de données de publications scientifiques issues de SCOPUS et utilise des modèles de langage pour fournir des informations fondées sur ces publications.

## Comment lancer l'application ?

### Via Streamlit

L'application est disponible en ligne à l'adresse suivante : [StreamlitApp](https://learning-analytics.streamlit.app/).

### En Local

1. Clonez le dépôt.
2. Récupérez votre clé API depuis Google AI Studio.
3. Créez un dossier `streamlit` et à l'intérieur, un fichier `secrets.toml`. Ajoutez votre clé API comme suit :
   ```toml
   GOOGLE_API_KEY="votre_clé_API"
   ```
4. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   ```
5. Activez l'environnement virtuel :
   - Sur Linux/Mac :
     ```bash
     source venv/bin/activate
     ```
   - Sur Windows :
     ```bash
     .\venv\Scripts\activate
     ```
6. Installez les dépendances nécessaires avec :
   ```bash
   pip install -r requirements.txt
   ```
7. Démarrez l'application avec :
   ```bash
   streamlit run app.py
   ```
