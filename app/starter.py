"""
Script pour démarrer l'application Streamlit avec une configuration qui évite les erreurs PyTorch.
"""
import os
import sys
import subprocess

def run_streamlit():
    """Exécute l'application Streamlit avec les paramètres qui évitent les erreurs PyTorch/asyncio."""
    # Désactiver la surveillance des modules dans Streamlit
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = ""
    os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Chemin vers le script app.py
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    print("Démarrage de l'application Streamlit...")
    print("Accédez à l'application dans votre navigateur à l'adresse: http://localhost:8501")
    print("Utilisez Ctrl+C pour arrêter l'application.")
    
    # Lancer Streamlit avec tous les arguments de ligne de commande qui pourraient être passés à ce script
    # en ignorant le nom du script lui-même
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true", 
           "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
    
    # Ajouter tous les arguments supplémentaires
    cmd.extend(sys.argv[1:])
    
    # Exécuter la commande
    subprocess.run(cmd)

if __name__ == "__main__":
    run_streamlit()