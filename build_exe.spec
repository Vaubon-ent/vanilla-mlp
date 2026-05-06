# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# Chemin vers le dossier du projet
# Dans un fichier .spec, on utilise le répertoire de travail actuel
project_root = Path(os.path.abspath('.'))

block_cipher = None

# Collecter toutes les données et DLL de PyTorch
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

# Collecter le module mnist (python-mnist)
try:
    mnist_datas, mnist_binaries, mnist_hiddenimports = collect_all('mnist')
except:
    mnist_datas, mnist_binaries, mnist_hiddenimports = [], [], []

a = Analysis(
    [str(project_root / 'src' / 'main.py')],
    pathex=[str(project_root / 'src')],  # Ajouter src au PYTHONPATH
    binaries=torch_binaries + mnist_binaries,  # Inclure les DLL de PyTorch et mnist
    datas=torch_datas + mnist_datas + [  # Inclure les données de PyTorch, mnist + nos données
        # Inclure le dossier models/ avec tous les fichiers .pt
        (str(project_root / 'models'), 'models'),
        # Inclure le dossier data/ avec les données MNIST
        (str(project_root / 'data'), 'data'),
    ],
    hiddenimports=torch_hiddenimports + mnist_hiddenimports + [  # Inclure les imports cachés de PyTorch, mnist + nos imports
        'ui',
        'ui.app',
        'ui.qwidget',
        'neural_network',
        'utils',
        'utils.mnist',
        'utils.resize_image',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'numpy',
        'scipy',
        'mnist',  # Module python-mnist
        'mnist.loader',  # Sous-module de python-mnist
        'kagglehub',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Mode onedir (dossier) au lieu de onefile pour PyTorch
# Les DLL restent dans le dossier et peuvent être chargées correctement
# Créer l'exe dans le mode onedir
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Les binaires seront dans le dossier, pas dans l'exe
    name='vanilla-mlp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Garder la console pour voir les messages de debug
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Vous pouvez ajouter un fichier .ico ici si vous en avez un
)

# Collecter tous les fichiers dans un dossier
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='vanilla-mlp',
)

