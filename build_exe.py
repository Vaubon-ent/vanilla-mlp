"""
Script pour créer un exécutable Windows avec PyInstaller.
Usage: python build_exe.py
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Vérifier que PyInstaller est installé
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller n'est pas installé. Installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Chemin vers le fichier .spec
    spec_file = Path(__file__).parent / "build_exe.spec"
    
    if not spec_file.exists():
        print(f"Erreur: Le fichier {spec_file} n'existe pas.")
        return 1
    
    # Vérifier que le modèle existe
    model_dir = Path(__file__).parent / "models"
    if not model_dir.exists() or not list(model_dir.glob("*.pt")):
        print("ATTENTION: Aucun modèle .pt trouvé dans le dossier models/")
        print("L'application fonctionnera mais ne pourra pas charger de modèle.")
        response = input("Continuer quand même ? (o/n): ")
        if response.lower() != 'o':
            return 1
    
    # Vérifier que les données MNIST existent
    data_dir = Path(__file__).parent / "data" / "mnist_data"
    if not data_dir.exists():
        print("ATTENTION: Le dossier data/mnist_data/ n'existe pas.")
        print("L'application ne pourra pas charger les images de test.")
        response = input("Continuer quand même ? (o/n): ")
        if response.lower() != 'o':
            return 1
    
    # Lancer PyInstaller
    print("=" * 70)
    print("Création de l'exécutable avec PyInstaller...")
    print("=" * 70)
    
    cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"]
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 70)
        print("OK - Executable cree avec succes !")
        print("=" * 70)
        print(f"L'executable se trouve dans: {Path(__file__).parent / 'dist' / 'vanilla-mlp' / 'vanilla-mlp.exe'}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERREUR lors de la creation de l'executable: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

