import sys, os, glob
from pathlib import Path
from neural_network import NeuralNetwork

# Ajouter src/ au PYTHONPATH pour les imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import *
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import *
from ui.qwidget import DrawingWidget


class MyApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.nn = NeuralNetwork()
        self.model_loaded = False  # Flag pour savoir si le modèle est chargé

        # Charger le modèle au démarrage (une seule fois)
        self.load_model_if_available()

        self.set_layout()
        self.init_canva()
        self.init_send_btn()
        # Ajouter le container du bouton après l'avoir créé
        self.main_layout.addWidget(self.button_container)
    # END FUNCTION

    def run(self):
        self.exec()
    # END FUNCTION

    def set_layout(self):
        self.window = QMainWindow()
        # Créer le widget central avec un layout vertical
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)  # Marges autour du layout
        self.main_layout.setSpacing(10)  # Espacement entre les widgets
        self.central_widget.setLayout(self.main_layout)
        # Ajout du layout a la window
        self.window.setCentralWidget(self.central_widget)
        # Force size de la fenêtre (560 pour le canvas + ~60 pour le bouton + marges)
        self.window.setMinimumSize(QSize(600, 650))
        self.window.show()
    # END FUNCTION

    def init_canva(self):
        # Canvas en haut (taille fixe)
        self.canva = DrawingWidget()
        self.canva.setFixedSize(560, 560)
        # Ajouter le canvas avec un alignement centré
        self.main_layout.addWidget(self.canva, alignment=Qt.AlignCenter)
    # END FUNCTION

    def init_send_btn(self):
        # Container pour le bouton centré en bas
        self.button_container = QWidget()
        self.button_container.setFixedHeight(50)  # Hauteur fixe pour le container
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Pas de marges dans le container
        self.button_container.setLayout(self.button_layout)

        self.button_layout.addStretch()
        self.reset_button = QPushButton("Send")
        self.reset_button.setMinimumSize(QSize(100, 40))  # Taille minimale du bouton
        self.reset_button.pressed.connect(self.send_image)
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addStretch()
    # END FUNCTION

    def send_image(self):
        draw = self.canva.save_draw()  # Utiliser save_draw() au lieu de resize()
        self.canva.refresh()

        if self.model_loaded:
            # run("USER") retourne déjà (predicted_label, probabilities)
            predicted_label, probabilities = self.nn.run("USER", user_image=draw)
            self.prediction_dlg(predicted_label, probabilities)
        else:
            # Afficher un message d'erreur si le modèle n'est pas disponible
            msg = QMessageBox(self.window)
            msg.setWindowTitle("Erreur")
            msg.setText("Aucun modèle disponible. Veuillez d'abord entraîner le modèle.")
            msg.exec()
    # END FUNCTION

    def load_model_if_available(self):
        """
        Charge le modèle une seule fois au démarrage si disponible.
        """
        models_dir = "models"
        model_files = []
        
        if os.path.exists(models_dir):
            # Chercher tous les fichiers model_*.pt
            model_files = glob.glob(os.path.join(models_dir, "model_*.pt"))
        
        if model_files:
            # Trier par date de modification (le plus récent en dernier)
            model_files.sort(key=lambda x: os.path.getmtime(x))
            latest_model = model_files[-1]
            
            print(f"Modèle sauvegardé trouvé: {latest_model}")
            print("Chargement du modèle...")
            
            # Charger le modèle
            self.nn.load_model(latest_model)
            self.model_loaded = True
            print("Modèle chargé avec succès.")
        else:
            print("Aucun modèle sauvegardé trouvé. Le modèle sera entraîné au premier envoi.")
            self.model_loaded = False
    # END FUNCTION
        
    def prediction_dlg(self, label: int, probabilities):
        dlg = QDialog(self.window)
        dlg.setWindowTitle("Prédiction")
        dlg.setMinimumSize(QSize(300, 200))
        
        layout = QVBoxLayout()
        
        # Afficher le label prédit
        label_text = QLabel(f"Chiffre prédit: <b>{label}</b>")
        label_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_text)
        
        # Afficher les probabilités
        prob_text = "Probabilités:\n"
        for i, prob in enumerate(probabilities):
            prob_text += f"{i}: {prob*100:.2f}%\n"
        
        prob_label = QLabel(prob_text)
        prob_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(prob_label)
        
        # Bouton OK
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dlg.accept)
        layout.addWidget(ok_button)
        
        dlg.setLayout(layout)
        dlg.exec()
    # END FUNCTION