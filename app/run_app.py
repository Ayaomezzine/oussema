import os
import sys
import warnings

# Ignorer les avertissements qui pourraient perturber l'exécution
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration pour éviter les problèmes avec PyTorch et Streamlit
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Désactiver complètement la surveillance des modules
os.environ['STREAMLIT_DISABLE_WATCHER'] = 'true'

print("Démarrage de l'application avec des paramètres optimisés pour Python 3.12...")
# Exécuter l'application Streamlit
os.system(f"{sys.executable} -m streamlit run app.py --server.headless=true --server.fileWatcherType=none")