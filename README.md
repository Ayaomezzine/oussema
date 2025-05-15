# Assistant Fondations Superficielles

Une application Streamlit qui permet de poser des questions sur les fondations superficielles et obtenir des réponses basées sur les documents PDF fournis, avec visualisation des images pertinentes.

## Fonctionnalités

- Interface de chat conviviale
- Recherche sémantique dans les documents PDF
- Affichage des extraits pertinents et des images associées
- Optimisé pour les appareils mobiles
- Entièrement en français

## Installation

1. Assurez-vous d'avoir Python 3.8+ installé
2. Installez les dépendances:

```bash
pip install -r app/requirements.txt
```

3. Pour Windows, vous devez installer Poppler pour que pdf2image fonctionne:
   - Téléchargez la dernière version de Poppler pour Windows: https://github.com/oschwartz10612/poppler-windows/releases/
   - Extrayez-le quelque part sur votre machine
   - Ajoutez le dossier `bin` de Poppler à votre PATH système

## Exécution

Pour lancer l'application:

```bash
streamlit run app/app.py
```

L'application sera accessible sur votre navigateur à l'adresse `http://localhost:8501`.

## Utilisation sur mobile

Pour accéder à l'application depuis un appareil mobile:
1. Exécutez l'application sur votre ordinateur
2. Trouvez l'adresse IP de votre ordinateur (ex: 192.168.1.x)
3. Sur votre appareil mobile, accédez à `http://[ADRESSE_IP]:8501`

**Note**: Votre ordinateur et votre appareil mobile doivent être sur le même réseau local.