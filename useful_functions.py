import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
import os
import random
from PIL import Image
from IPython.display import display
import IPython.display as ipd

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]  # moyennes des canaux RGB de l'image 


def display_images_side_by_side(img_path1, img_path2, titles=None):
    """
    Affiche deux images côte à côte dans un notebook Jupyter avec des légendes sous chaque image.
    
    Arguments:
        img_path1 (str): Chemin vers la première image.
        img_path2 (str): Chemin vers la deuxième image.
        titles (list, optional): Titres pour les images. Exemple : ["Image 1", "Image 2"].
    """
    # Charger et convertir les images en RGB
    img1 = cv.imread(img_path1)
    img2 = cv.imread(img_path2)
    
    if img1 is None:
        raise FileNotFoundError(f"Image non trouvée : {img_path1}")
    if img2 is None:
        raise FileNotFoundError(f"Image non trouvée : {img_path2}")
    
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    
    # Créer une figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Afficher la première image
    axs[0].imshow(img1_rgb)
    axs[0].axis('off')
    if titles and len(titles) > 0:
        axs[0].set_title(titles[0])
    axs[0].set_xlabel("Image d'origine", fontsize=12, ha='center')  # Légende sous la première image
    
    # Afficher la deuxième image
    axs[1].imshow(img2_rgb)
    axs[1].axis('off')
    if titles and len(titles) > 1:
        axs[1].set_title(titles[1])
    axs[1].set_xlabel("Image après pré-traitement", fontsize=12, ha='center')  # Légende sous la deuxième image
    
    # Ajuster les éléments et afficher les images
    plt.tight_layout()
    plt.show()



def select_files(directory):
    """
    Selects the second file, the last file, and three other random files from the given directory.
    Returns two sorted lists:
    1. A list of full file paths sorted by filename.
    2. A list of iteration strings sorted numerically.
    """
    # Lister et filtrer uniquement les fichiers
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Vérifier qu'il y a assez de fichiers
    if len(files) < 5:
        raise ValueError("Not enough files in the directory.")

    # Trier les fichiers par nom (ordre alphabétique)
    files.sort()

    # Sélectionner les fichiers : deuxième et dernier
    selected_files = {files[1], files[-1]}  # Utilisation d'un set pour éviter les doublons

    # Sélectionner aléatoirement 3 autres fichiers
    remaining_files = list(set(files) - selected_files)
    selected_files.update(random.sample(remaining_files, 3))

    # Convertir en liste et trier numériquement
    selected_files = sorted(selected_files, key=lambda x: int(x.split('.')[0]))

    # Construire les chemins complets
    file_paths = [os.path.join(directory, f) for f in selected_files]

    # Construire la liste des iterations
    iterations_list = sorted([f"{int(f.split('.')[0])} iterations" for f in selected_files], key=lambda x: int(x.split()[0]))
    
    return file_paths, iterations_list





def display_five_images(image_paths, captions):
    """
    Affiche 5 images côte à côte avec leurs légendes.

    Args:
        image_paths (list): Liste des chemins des 5 images.
        captions (list): Liste des légendes associées aux images.

    """
    if len(image_paths) != 5 or len(captions) != 5:
        raise ValueError("Vous devez fournir exactement 5 images et 5 légendes.")

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for i, ax in enumerate(axes):
        img = mpimg.imread(image_paths[i])  # Charger l'image
        ax.imshow(img)
        ax.set_title(captions[i])  # Ajouter la légende
        ax.axis("off")  # Cacher les axes

    plt.show()


def generate_out_img_name(config, img_id):
    '''
    Generate a name for the output image.
    Example: 'c1-s1.jpg'
    where c1: content_img_name, and
          s1: style_img_name.
    '''
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    suffix = f'{config["img_format"][1]}'
    return f"{prefix}_{img_id}{suffix}"

def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations):
    '''
    Save the generated image.
    If saving_freq == -1, only the final output image will be saved.
    Else, intermediate images can be saved too.
    '''
    saving_freq = 25
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    if (img_id == num_of_iterations-1) or (img_id % saving_freq == 0 and saving_freq != -1):
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config, img_id)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))   #On applique une dénormalisation pour l'affichage
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
    
    


def generate_difference_heatmap(img1_path, img2_path):
    """
    Génère une heatmap de différence entre deux images.

    Arguments:
        img1_path (str): Chemin de la première image.
        img2_path (str): Chemin de la deuxième image.
        output_heatmap_path (str, optionnel): Si fourni, sauvegarde la heatmap sous ce chemin.
    
    Retourne:
        heatmap (numpy array): La heatmap normalisée.
    """
    # Charger les images
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Une ou les deux images n'ont pas été trouvées.")
    
    # Vérifier que les dimensions sont identiques
    if img1.shape != img2.shape:
        raise ValueError("Les dimensions des images ne correspondent pas.")
    
    # Calculer la différence pixel à pixel (dans l'espace RGB)
    diff = cv.absdiff(img1, img2)  # Différence absolue pour chaque canal (R, G, B)
    
    # Optionnel : convertir la différence en intensité unique (niveaux de gris)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    
    # Normaliser les valeurs pour qu'elles soient entre 0 et 255
    heatmap = cv.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    return heatmap

# Exemple de visualisation
def show_heatmap(heatmap):
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot')  # 'hot' pour une heatmap colorée
    plt.axis('off')
    plt.show()



def display_generated_image(content_image, style_image, num_of_iterations):
    """
    Affiche l'image de contenu, l'image de style et l'image générée côte à côte.

    :param content_image: Nom du fichier de l'image de contenu (avec extension).
    :param style_image: Nom du fichier de l'image de style (avec extension).
    :param num_of_iterations: Nombre d'itérations de l'optimisation.
    """
    def load_image(image_path):
        """ Charge une image si elle existe, sinon retourne None. """
        if os.path.exists(image_path):
            img = cv.imread(image_path)
            return cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convertir BGR → RGB
        else:
            print(f"L'image {image_path} n'existe pas.")
            return None

    # Chargement des images
    content_image_path = os.path.join("./data/content-images", content_image)
    style_image_path = os.path.join("./data/style-images", style_image)

    content_img = load_image(content_image_path)
    style_img = load_image(style_image_path)

    # Construction du chemin de l'image générée
    content_name = os.path.splitext(content_image)[0]
    style_name = os.path.splitext(style_image)[0]
    folder_name = f"combined_{content_name}_{style_name}"
    folder_path = os.path.join("./data/output-images", folder_name)
    generated_image_path = os.path.join(folder_path, f"{num_of_iterations:04d}.jpg")

    generated_img = load_image(generated_image_path)

    # Affichage des images si elles existent
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [content_img, style_img, generated_img]
    titles = ["Image de Contenu", "Image de Style", f"Image Générée ({num_of_iterations} itérations)"]

    for ax, img, title in zip(axes, images, titles):
        if img is not None:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.show()


def create_gif_from_images(input_directory, output_gif_path, duration=200):
    """
    Crée un GIF animé à partir d'images dans un répertoire.

    Arguments :
        input_directory (str) : Le chemin du répertoire contenant les images.
        output_gif_path (str) : Le chemin où sauvegarder le GIF.
        duration (int) : Durée d'affichage de chaque image dans le GIF, en millisecondes.

    Retourne :
        None
    """
    # Lister les fichiers d'images dans le répertoire
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    if not image_files:
        raise ValueError(f"Aucune image trouvée dans le répertoire : {input_directory}")
    
    # Trier les images par nom pour garantir l'ordre
    image_files.sort()

    # Charger les images
    images = [Image.open(os.path.join(input_directory, img)) for img in image_files]

    # Vérifier que toutes les images ont la même taille (nécessaire pour créer un GIF)
    sizes = [image.size for image in images]
    if len(set(sizes)) > 1:
        raise ValueError("Toutes les images doivent avoir la même taille pour créer un GIF.")
    
    # Créer et sauvegarder le GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF créé avec succès : {output_gif_path}")


def display_gif(gif_path):
    """
    Affiche un GIF dans un notebook Jupyter.

    Args:
        gif_path (str): Chemin du GIF à afficher.
    """
    display(ipd.Image(gif_path))