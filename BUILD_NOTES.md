# Notes sur la création de l'exécutable

## Problème actuel

PyTorch ne peut pas charger ses DLL dans un exe PyInstaller. L'erreur indique :
```
OSError: [WinError 1114] Une routine d'initialisation d'une bibliothèque de liens dynamiques (DLL) a échoué. 
Error loading "C:\Users\sam97\AppData\Local\Temp\_MEI31162\torch\lib\c10.dll" or one of its dependencies.
```

## Solutions possibles

### Solution 1 : Mode onedir (recommandé)
Le fichier `build_exe.spec` est configuré pour créer un dossier au lieu d'un seul exe. 
**IMPORTANT** : Vous devez reconstruire l'exe après chaque modification du .spec :

```bash
python build_exe.py
```

Le résultat sera dans `dist/vanilla-mlp/` (un dossier, pas un seul fichier).

### Solution 2 : Installer Visual C++ Redistributables
PyTorch nécessite les Visual C++ Redistributables. Téléchargez et installez :
- Visual C++ Redistributable 2015-2022 (x64)
- Disponible sur le site Microsoft

### Solution 3 : Utiliser PyTorch CPU uniquement
Si vous n'avez pas besoin de GPU, vous pouvez installer PyTorch CPU-only qui a moins de dépendances DLL.

### Solution 4 : Alternative - Ne pas créer d'exe
Au lieu de créer un exe, vous pourriez :
1. Créer un script batch (.bat) qui active l'environnement virtuel et lance l'app
2. Utiliser cx_Freeze au lieu de PyInstaller
3. Créer un installer avec Inno Setup ou NSIS qui installe Python et les dépendances

## Vérification

Pour vérifier que le mode onedir est utilisé :
1. Après la construction, vérifiez que `dist/vanilla-mlp/` est un **dossier** avec plusieurs fichiers
2. Si vous voyez seulement `dist/vanilla-mlp.exe` (un seul fichier), le mode onefile est encore utilisé

## Distribution

En mode onedir, vous devez distribuer **tout le dossier** `vanilla-mlp/`, pas seulement l'exe.

