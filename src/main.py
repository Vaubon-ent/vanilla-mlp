from ui.app import MyApp
from neural_network import NeuralNetwork
import os, glob

if __name__ == "__main__":
    app = MyApp()
    app.run()

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
