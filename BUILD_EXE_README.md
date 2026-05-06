# Instructions pour créer un exécutable (.exe)

Ce guide explique comment créer un exécutable Windows de l'application avec PyInstaller, incluant le modèle .pt et les données MNIST.

## Prérequis

1. **Installer PyInstaller** :
```bash
pip install pyinstaller
```

2. **Vérifier que vous avez** :
   - Un modèle entraîné dans `models/model_*.pt`
   - Les données MNIST dans `data/mnist_data/`

## Création de l'exécutable

### Méthode 1 : Utiliser le script automatique (recommandé)

```bash
python build_exe.py
```

Le script va :
- Vérifier que PyInstaller est installé
- Vérifier la présence du modèle et des données
- Créer l'exécutable avec toutes les dépendances

### Méthode 2 : Utiliser directement PyInstaller

```bash
pyinstaller build_exe.spec --clean
```

## Résultat

L'exécutable sera créé dans le dossier `dist/vanilla-mlp.exe`

## Structure de l'exécutable

L'exécutable inclut :
- ✅ Toutes les dépendances Python (PyQt5, PyTorch, numpy, scipy, etc.)
- ✅ Le dossier `models/` avec votre modèle .pt
- ✅ Le dossier `data/` avec les données MNIST
- ✅ Tous les fichiers source nécessaires

## Notes importantes

1. **Taille de l'exécutable** : L'exécutable sera assez volumineux (~500MB-1GB) car il inclut PyTorch et toutes ses dépendances.

2. **Console** : L'exécutable affiche une console pour les messages de debug. Pour la masquer, modifiez `console=True` en `console=False` dans `build_exe.spec`.

3. **Icône** : Vous pouvez ajouter une icône en modifiant `icon=None` dans `build_exe.spec` et en spécifiant le chemin vers un fichier `.ico`.

4. **Déploiement** : Vous pouvez distribuer uniquement le fichier `.exe` - tout est inclus dedans.

## Dépannage

### Erreur "Module not found"
Si vous obtenez une erreur de module manquant, ajoutez-le dans `hiddenimports` dans `build_exe.spec`.

### Le modèle n'est pas trouvé
Vérifiez que le dossier `models/` contient au moins un fichier `model_*.pt` avant de créer l'exe.

### Les données MNIST ne sont pas trouvées
Vérifiez que le dossier `data/mnist_data/` existe et contient les fichiers de test.

## Pour créer un exe sans console

Modifiez `build_exe.spec` ligne 50 :
```python
console=False,  # Au lieu de console=True
```

## Pour créer un exe avec une icône

1. Créez ou trouvez un fichier `.ico`
2. Placez-le dans le dossier du projet
3. Modifiez `build_exe.spec` ligne 54 :
```python
icon='votre_icone.ico',  # Au lieu de icon=None
```

