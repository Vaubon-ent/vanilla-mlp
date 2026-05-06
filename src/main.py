import os, glob
import sys
import traceback
from datetime import datetime

# Fonction pour écrire dans un fichier de log
def log_error(message):
    """Écrit un message d'erreur dans un fichier log."""
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'error.log')
        log_file = os.path.abspath(log_file)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except:
        pass  # Si on ne peut pas écrire le log, on continue

# Configurer sys.excepthook pour capturer TOUTES les erreurs, même celles non gérées
def exception_handler(exc_type, exc_value, exc_traceback):
    """Gère toutes les exceptions non capturées."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_msg = f"\n{'=' * 70}\nERREUR CRITIQUE NON GÉRÉE\n{'=' * 70}\n"
    error_msg += f"Type d'erreur: {exc_type.__name__}\n"
    error_msg += f"Message: {str(exc_value)}\n\n"
    error_msg += "Traceback complet:\n"
    error_msg += ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    error_msg += f"\n{'=' * 70}\n"
    
    print(error_msg)
    log_error(error_msg)
    
    # Pause pour permettre de lire l'erreur
    try:
        input("\nAppuyez sur Entrée pour fermer...")
    except:
        import time
        time.sleep(10)  # Attendre 10 secondes si input() ne fonctionne pas
    sys.exit(1)

# Installer le gestionnaire d'exceptions global
sys.excepthook = exception_handler

# Ajouter src/ au PYTHONPATH pour les imports (nécessaire pour PyInstaller)
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

if __name__ == "__main__":
    try:
        print("=" * 70)
        print("DÉMARRAGE DE L'APPLICATION")
        print("=" * 70)
        print(f"Python: {sys.version}")
        print(f"Chemin de travail: {os.getcwd()}")
        print(f"Chemin src: {src_path}")
        print("-" * 70)
        
        print("Import des modules...")
        from ui.app import MyApp
        from neural_network import NeuralNetwork
        print("✓ Modules importés avec succès.")
        
        print("Création de l'application...")
        app = MyApp()
        print("✓ Application créée avec succès.")
        
        print("Lancement de l'interface...")
        app.run()
        
        print("\nApplication fermée normalement.")
    except Exception as e:
        # Afficher l'erreur complète
        error_msg = f"\n{'=' * 70}\nERREUR CRITIQUE\n{'=' * 70}\n"
        error_msg += f"Type d'erreur: {type(e).__name__}\n"
        error_msg += f"Message: {str(e)}\n\n"
        error_msg += "Traceback complet:\n"
        error_msg += ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_msg += f"\n{'=' * 70}\n"
        
        print(error_msg)
        log_error(error_msg)
        
        # Pause pour permettre de lire l'erreur
        try:
            input("\nAppuyez sur Entrée pour fermer...")
        except:
            import time
            time.sleep(10)  # Attendre 10 secondes si input() ne fonctionne pas
        sys.exit(1)

    # models_dir = "models"
    # model_files = []
    # nn = NeuralNetwork()
    
    # if os.path.exists(models_dir):
    #     # Chercher tous les fichiers model_*.pt
    #     model_files = glob.glob(os.path.join(models_dir, "model_*.pt"))
    
    # if model_files:
    #     # Trier par date de modification (le plus récent en dernier)
    #     model_files.sort(key=lambda x: os.path.getmtime(x))
    #     latest_model = model_files[-1]
        
    #     print(f"Modèle sauvegardé trouvé: {latest_model}")
    #     print("Chargement du modèle...")
        
    #     # Charger le modèle
    #     nn.load_model(latest_model)
    #     model_loaded = True
    #     print("Modèle chargé avec succès.")
    #     nn.run("TEST_MNIST")
    # else:
    #     print("Aucun modèle sauvegardé trouvé. Le modèle sera entraîné au premier envoi.")
    #     model_loaded = False
