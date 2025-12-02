from scipy.ndimage import zoom
from PyQt5.QtGui import QImage
import numpy as np


def display_image_ascii(image_normalized, width=28, threshold=0.5):
    """
    Affiche une image normalisée (0-1) en ASCII art, similaire à mndata.display()
    
    Args:
        image_normalized: Liste de 784 pixels normalisés entre 0 et 1
        width: Largeur de l'image (28 pour MNIST)
        threshold: Seuil pour déterminer si un pixel est noir ou blanc (0.5 = 127.5/255)
    
    Returns:
        String ASCII représentant l'image
    """
    render = ''
    for i, pixel in enumerate(image_normalized):
        if i % width == 0:
            render += '\n'
        # Dans MNIST normalisé : 0.0 = noir (chiffre), 1.0 = blanc (fond)
        # On inverse pour l'affichage : si pixel < threshold, c'est du noir (chiffre) -> '@'
        # Si pixel >= threshold, c'est du blanc (fond) -> '.'
        if pixel < threshold:
            render += '@'  # Pixel sombre (chiffre)
        else:
            render += '.'  # Pixel clair (fond)
    return render

def resize_image(image: QImage):
    # Grayscale convertion
    resized_image = image.convertToFormat(QImage.Format_Grayscale8)

    # Convertion numpy array
    width = resized_image.width()
    height = resized_image.height()
    ptr = resized_image.bits()
    ptr.setsize(resized_image.byteCount())
    arr = np.array(ptr).reshape(height, width)
    
    # Debug : vérifier les valeurs avant redimensionnement
    print(f"Image originale: {width}x{height}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    print(f"Pixels noirs (<50): {(arr < 50).sum()}, Pixels blancs (>200): {(arr > 200).sum()}")

    # Réduction vectorisée : facteur de réduction = 560/28 = 20
    # On reshape l'image en blocs de 20x20 et on prend le minimum de chaque bloc
    factor = int(height / 28)  # Devrait être 20
    
    # S'assurer que l'image est divisible par le facteur (tronquer si nécessaire)
    h_trunc = (height // factor) * factor
    w_trunc = (width // factor) * factor
    arr = arr[:h_trunc, :w_trunc]
    
    # Reshape en (28, 20, 28, 20) pour créer des blocs de 20x20
    # Puis prendre le minimum sur les axes 1 et 3 (les dimensions du bloc)
    arr_reshaped = arr.reshape(28, factor, 28, factor)
    arr_28 = arr_reshaped.min(axis=(1, 3)).astype(np.uint8)
    
    # Debug : vérifier les valeurs après sous-échantillonnage
    print(f"Image 28x28: min={arr_28.min()}, max={arr_28.max()}, mean={arr_28.mean():.1f}")
    print(f"Pixels noirs (<50): {(arr_28 < 50).sum()}, Pixels blancs (>200): {(arr_28 > 200).sum()}")

    # Normaliser entre 0 et 1 (comme format_images dans mnist.py)
    # Dans MNIST : chiffres noirs (0) -> 0.0, fond blanc (255) -> 1.0
    arr_28_normalized = arr_28.astype(np.float32) / 255.0
    
    # Debug : vérifier si l'image est inversée (trop de blanc = mean proche de 1.0)
    mean_normalized = arr_28_normalized.mean()
    if mean_normalized > 0.9:
        print(f"ATTENTION: Image très claire (mean={mean_normalized:.3f}), peut-être inversée ?")
        print(f"Tentative d'inversion des couleurs...")
        # Inverser les couleurs : 1.0 - pixel (noir devient blanc, blanc devient noir)
        arr_28_normalized = 1.0 - arr_28_normalized
        print(f"Après inversion: mean={arr_28_normalized.mean():.3f}")

    # Aplatir en vecteur de 784 pixels (comme attendu par le MLP)
    arr_flat = arr_28_normalized.flatten()

    return arr_flat.tolist()  # Retourner une liste comme format_images()

