# Vanilla MLP — Neural Network from Scratch

Implémentation d'un réseau de neurones multicouche (MLP) **from scratch** en Python/PyTorch, entraîné sur le dataset MNIST (classification de chiffres manuscrits), avec une interface graphique PyQt5.

---

## Objectif

Comprendre en profondeur le fonctionnement d'un réseau de neurones en implémentant manuellement :
- La propagation avant (forward pass)
- Le calcul du gradient (backpropagation)
- La descente de gradient avec learning rate decay
- Le chargement et preprocessing des données MNIST
- Un entraînement multi-threadé avec verrou

---

## Architecture

```
src/
├── main.py              # Point d'entrée, gestion des erreurs globales
├── neural_network.py    # Classe NeuralNetwork — cœur du modèle
├── ui/
│   ├── app.py           # Interface PyQt5
│   └── qwidget.py       # Composants UI
└── utils/
    └── mnist.py         # Chargement du dataset MNIST
```

---

## Stack technique

| Outil | Usage |
|---|---|
| Python 3.x | Langage principal |
| PyTorch | Tenseurs + calcul GPU/CPU |
| NumPy | Opérations matricielles |
| PyQt5 | Interface graphique |
| MNIST | Dataset d'entraînement (chiffres manuscrits 28×28px) |

**Compatibilité GPU :** CUDA (NVIDIA) et ROCm 5.7 (AMD)

---

## Lancer le projet

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

> Pour GPU AMD : `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7`

### 2. Télécharger le dataset MNIST

Le dataset n'est pas inclus dans le repo. Le charger via torchvision :

```python
from torchvision import datasets
datasets.MNIST('./data', train=True, download=True)
datasets.MNIST('./data', train=False, download=True)
```

### 3. Lancer

```bash
python src/main.py
```

---

## Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Learning rate initial | 0.01 |
| Learning rate minimum | 0.001 |
| Decay rate | 0.95 |
| Batch size | 500 |
| Epochs max | 10 |

---

## Ce que j'ai appris

- Implémentation manuelle de la backpropagation sans autograd
- Gestion du multi-threading pour l'entraînement (Lock, Thread)
- Détection dynamique CPU/GPU (CUDA + ROCm)
- Packaging d'une application Python en exécutable Windows (PyInstaller + PyQt5)
- Problématiques DLL avec PyTorch en mode bundle

---

*Projet personnel d'exploration — Mai 2026*
