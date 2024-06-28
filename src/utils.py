
import random
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from datasets import DatasetDict
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
from PIL import Image


def count_images_per_split(dataset: DatasetDict) -> dict:
    """
    Funktion, um die Anzahl der Bilder pro Split (Trainings-, Validierungs-, Testsplit) in einem Datensatz zu zählen.

    Args:
    - dataset: DatasetDict-Objekt, das die Splits train, validation, test enthält.

    Returns:
    - image_counts: Dictionary mit der Anzahl der Bilder pro Split.
    """
    image_counts = {
        "train": len(dataset["train"]),
        "validation": len(dataset["validation"]),
        "test": len(dataset["test"])
    }
    return image_counts


def find_unique_labels(dataset: DatasetDict, column: str = "doc_category") -> list:
    """
    Funktion, um alle eindeutigen Labels in der angegebenen Spalte über alle Splits eines Datensatzes zu finden.

    Args:
    - dataset: DatasetDict-Objekt, das die Splits train, validation, test enthält.
    - column: Name der Spalte, die die Labels enthält. Standardmäßig "doc_category".

    Returns:
    - unique_labels: Liste der eindeutigen Labels, die in der angegebenen Spalte über alle Splits gefunden wurden.
    """
    unique_labels = []
    for split in dataset.keys():
        unique_labels.extend(dataset[split][column])
    unique_labels = list(set(unique_labels))
    return unique_labels


def count_labels_per_split(dataset: DatasetDict, column: str = "doc_category") -> dict:
    """
    Funktion, um die Anzahl der Labels pro Split (Trainings-, Validierungs-, Testsplit) in einem Datensatz zu zählen.

    Args:
    - dataset: DatasetDict-Objekt, das die Splits train, validation, test enthält.
    - column: Name der Spalte, die die Labels enthält. Standardmäßig "doc_category".

    Returns:
    - label_counts: Dictionary mit der Anzahl der Labels pro Split.
    """
    label_counts = {
        "train": {},
        "validation": {},
        "test": {}
    }
    for split in dataset.keys():
        labels = dataset[split][column]
        label_counts[split] = {label: labels.count(label) for label in set(labels)}
    return label_counts


def plot_label_counts(label_counts: dict):
    """
    Funktion zum Plotten der prozentualen Verteilung der Labels pro Split als horizontalen gestapelten Balkendiagramm mit Legende.

    Args:
    - label_counts: Dictionary mit der Anzahl der Labels pro Split.
    """
    # Transformieren der Daten in ein DataFrame für die Visualisierung
    df = pd.DataFrame(label_counts).T

    # Berechnung der prozentualen Verteilung
    df_percent = df.div(df.sum(axis=1), axis=0) * 100

    # Farbschema festlegen
    countbar_colors = ["#9DA2BE", "#737892", "#4A5068", "#252b41", "#B2B5CC", "#161a28"]

    # Plot mit Seaborn: Prozentuale Verteilung der Klassen je Split
    sns.set(style="white")
    plt.figure(figsize=(10, 6))
    
    # Erstellen des gestapelten Balkendiagramms
    ax = df_percent.plot(kind='barh', stacked=True, color=countbar_colors)

    # Anzeige der prozentualen Werte innerhalb der Balken, wenn größer als 10%
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if width > 10:
            x, y = p.get_xy()
            ax.text(x + width/2, y + height/2, f'{width:.1f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    # Legende anzeigen
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10)

    # Achsenbeschriftungen und Titel
    plt.ylabel("", fontsize=14)
    plt.xlabel("", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), color="#252b41", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), color="#252b41", fontsize=10)
    plt.title("Prozentuale Verteilung der Klassen je Split", fontsize=12, pad=20, color="#252b41", fontweight=550)

    # Sonstige Einstellungen
    ax.spines["left"].set_color("#252b41")
    ax.set(xticklabels=[])
    ax.xaxis.set_ticks_position("none")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)

    # Speichern und anzeigen
    plt.savefig("../reports/figures/Klassenverteilung_Datensatz_proz.png", dpi=300, bbox_inches="tight")
    plt.show()


def display_class_image_examples(dataset: List[Dict[str, Any]], num_images_per_class: int) -> None:
    """
    Diese Funktion zeigt eine festgelegte Anzahl von Bildern für verschiedene Klassen aus einem Huggingface-Datensatz an,
    sortiert nach der doc_category. Die Originalhöhe und -breite werden für jedes Bild aus dem Datensatz entnommen.

    Args:
        dataset (List[Dict[str, Any]]): Der Huggingface-Datensatz, der die Bilder und zugehörigen Label enthält.
        num_images_per_class (int): Die Anzahl der Bilder pro Klasse, die angezeigt und gespeichert werden sollen.

    Returns:
        None
    """
    # Alle einzigartigen doc_categories sammeln
    doc_categories = set([example["doc_category"] for example in dataset])

    # Durchlaufen der einzigartigen doc_categories und Anzeigen der Bilder für jede Kategorie
    for doc_category in sorted(doc_categories):
        # Initialisierung einer Liste für die ausgewählten Bilder und deren Größen
        selected_images = []

        # Sammeln von Bildern und Größen dieser doc_category
        for example in dataset:
            if len(selected_images) >= num_images_per_class:
                break
            if example["doc_category"] == doc_category:
                selected_images.append(example["image"])

        # Anzeigen und Speichern der Bilder für diese doc_category
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Dokumentenkategorie: {doc_category}", fontsize=16)
        
        for i, image in enumerate(selected_images):
            plt.subplot(1, num_images_per_class, i + 1)
            plt.imshow(image)
            plt.axis('off')

        # Speichern und anzeigen
        plt.subplots_adjust(wspace=-0.3)
        plt.tight_layout()
        plt.savefig(f"../reports/figures/Dokumentenkategorie_{doc_category}.png", dpi=300, bbox_inches="tight")
        plt.show()

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Diese Funktion dreht ein OpenCV-Bild um einen bestimmten Winkel und gibt das gedrehte Bild zurück.
    
    Args:
        image (np.ndarray): Das Eingangsbild als NumPy-Array.
        angle (float): Der Winkel, um den das Bild gedreht werden soll (in Grad).

    Returns:
        np.ndarray: Das gedrehte Bild als NumPy-Array.
    """
    # Bestimme die Mitte des Bildes
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Erstelle die Rotationsmatrix um die Mitte des Bildes
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Wende die Rotationsmatrix auf das Bild an
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(255,255,255))
    
    return rotated_image

def pil_to_cv(image: Image.Image) -> np.ndarray:
    """
    Konvertiert ein PIL-Bild in ein OpenCV-Bild.
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def from_cv_to_pil(image: np.ndarray) -> Image.Image:
    """
    Konvertiert ein Bild vom OpenCV-Format zurück ins PIL-Format.
    
    Args:
        cv_image (np.ndarray): Das Eingangsbild im OpenCV-Format.

    Returns:
        PIL.Image.Image: Das konvertierte Bild im PIL-Format.
    """
    # Wandele das Farbformat von BGR zu RGB um (für PIL)
    cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Konvertiere das Numpy-Array in ein PIL-Bild
    pil_image = Image.fromarray(cv_image)
    return pil_image
