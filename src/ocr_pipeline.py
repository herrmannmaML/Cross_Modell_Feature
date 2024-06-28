import re
import os
import cv2
import math
import logging
import warnings
from deskew import determine_skew
import urllib.request
from typing import Union
from skimage.morphology import thin
import numpy as np
import urllib.request
from PIL import Image
import nltk
from symspellpy import SymSpell, Verbosity
from spellchecker import SpellChecker as PySpellChecker
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import stanza
import pkg_resources

# zeige keine Warnungen an
warnings.filterwarnings("ignore")

# Logging-Config
logging.basicConfig(filename='../project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRPreprocessor:
    """
    Eine Klasse zur Vorverarbeitung von Bildern für die OCR.
    """

    def __init__(self, image: Union[np.ndarray, Image.Image]):
        """
        Initialisiert die OCRPreprocessor-Klasse mit einem Bild.
        
        Args:
            image (Union[np.ndarray, Image.Image]): Das Eingangsbild als NumPy-Array oder PIL.Image.Image.
                Wenn das Bild ein PIL-Bild ist, wird es in ein OpenCV-Bild konvertiert.
        
        Returns:
            None
        """
        if isinstance(image, Image.Image):
            self.image = self.pil_to_cv(image)
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise TypeError("Das Bild muss entweder vom Typ PIL.Image.Image oder numpy.ndarray sein.")

    def pil_to_cv(self, image: Image.Image) -> np.ndarray:
        """
        Konvertiert ein PIL-Bild in ein OpenCV-Bild.
        
        Args:
            image (Image.Image): Das Eingangsbild als PIL-Bild.
        
        Returns:
            np.ndarray: Das konvertierte Bild als NumPy-Array.
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def from_cv_to_pil(self, image: np.ndarray) -> None:
        """
        Konvertiert ein Bild vom OpenCV-Format zurück ins PIL-Format.
        
        Args:
            image (np.ndarray): Das Eingangsbild im OpenCV-Format.
        
        Returns:
            Image.Image: Das konvertierte Bild im PIL-Format.
        """
        # Wandele das Farbformat von BGR zu RGB um (für PIL)
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Konvertiere das Numpy-Array in ein PIL-Bild
        pil_image = Image.fromarray(cv_image)
        return pil_image
    
    def cropping(self, buffer_size: int = 10) -> None:
        """
        Entfernt weiße Bildränder mit entsprechendem Puffer.
        
        Args:
            buffer_size (int): Die Größe des Puffers für die Beschnittgrenzen.
        
        Returns:
            None
        """
        # Konvertieren des Bildes in Graustufen
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Festlegen einer Schwellenwert für binäre Umwandlung (Schwarzweiß)
        threshold = 200

        # Konvertieren des Bildes in ein binäres Schwarzweißbild
        _, binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY_INV)

        # Festlegen der Beschnittgrenzen
        x, y, w, h = cv2.boundingRect(binary_image)

        # Erweiterung der Beschnittgrenzen um den Puffer
        x = max(0, x - buffer_size)
        y = max(0, y - buffer_size)
        w = min(self.image.shape[1] - x, w + 2 * buffer_size)
        h = min(self.image.shape[0] - y, h + 2 * buffer_size)

        # Beschnitt des Bildes basierend auf den erweiterten Beschnittgrenzen
        self.image = self.image[y:y+h, x:x+w]

    def adjust_brightness_contrast(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        """
        Passt die Helligkeit und den Kontrast des Bildes an.
        
        Args:
            alpha (float): Der Verstärkungsfaktor für den Kontrast (Standardwert ist 1.0).
            beta (float): Der Bias-Wert für die Helligkeit (Standardwert ist 0.0).
        
        Returns:
            None
        """
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)

    def resize(self, factor: float) -> None:
        """
        Ändert die Größe des Bildes um einen gegebenen Faktor.
        
        Args:
            factor (float): Der Faktor, um den das Bild vergrößert oder verkleinert werden soll.
        
        Returns:
            None
        """
        self.image = cv2.resize(self.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    def upscale(self) -> None:
        """
        Ändert die Größe des Bildes durch ein Neuronales Netz.
        
        Returns:
            None
        """
        # Überprüfe, ob Modell für Upscaling bereits lokal vorhanden ist
        if os.path.isfile("../models/FSRCNN_x3.pb"):
            pass
        else:
            upscaling_model_url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb"
            upscaling_model_path = "../models/FSRCNN_x3.pb"
            logging.info(f"[INFO]: Modell für Upscaling wird von {upscaling_model_url} geladen und unter {upscaling_model_path} gespeichert")
            urllib.request.urlretrieve(upscaling_model_url, upscaling_model_path)
        
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("../models/FSRCNN_x3.pb")
        sr.setModel("fsrcnn",3)
        self.image = sr.upsample(self.image)

    def power_law_transform(self, gamma: float) -> None:
        """
        Führt eine Power-Law-Transformation auf dem Bild durch.
        
        Args:
            gamma (float): Der Gamma-Wert für die Transformation.
        
        Returns:
            None
        """
        self.image = np.array(255*(self.image / 255)**gamma, dtype = 'uint8')

    def to_gray(self) -> None:
        """
        Konvertiert das Bild in Graustufen.
        
        Returns:
            None
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def contrast_stretching(self) -> None:
        """
        Führt eine Kontrastdehnung auf dem Bild in Graustufen durch.
        
        Returns:
            None
        """
        minmax_img = np.zeros((self.image.shape[0],self.image.shape[1]),dtype = 'uint8')
        self.image = cv2.normalize(self.image, minmax_img, 0, 255, cv2.NORM_MINMAX)
    
    def sharpen(self, kernel_type: str = "laplace_standard") -> None:
        """
        Wendet einen Schärfungsfilter auf ein OpenCV-Bild an.
        
        Args:
            kernel_type (str): Der gewünschte Kernel-Typ für den Schärfungsfilter.
                               Mögliche Werte: "laplace_robust", "laplace_classic", "laplace_standard",
                               "laplace_diagonal", "sobel_horizontal", "sobel_vertical",
                               "prewitt_horizontal", "prewitt_vertical", "scharr_horizontal",
                               "scharr_vertical", "pil_standard"
        
        Returns:
            None
        """
        # Listw an validen Kernel-Types
        valid_kernel_types = ["laplace_robust", "laplace_classic", "laplace_standard",
                            "laplace_diagonal", "sobel_horizontal", "sobel_vertical",
                            "prewitt_horizontal", "prewitt_vertical", "scharr_horizontal",
                            "scharr_vertical", "pil_standard"]
        if kernel_type not in valid_kernel_types:
            logging.error(f"Ungültiger Schärfekernel '{kernel_type}' angegeben. Bitte wählen Sie zwischen {', '.join(valid_kernel_types)}.")
            raise ValueError("Ungültiger Kernel-Typ. Bitte wählen Sie einen der folgenden: "
                            + ", ".join(valid_kernel_types))
        
        if kernel_type == "laplace_robust":
            kernel = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])
        
        elif kernel_type == "pil_standard":
            kernel = np.array([[-2, -2, -2],
                               [-2,  32, -2],
                               [-2, -2, -2]])
        
        elif kernel_type == "laplace_classic":
            kernel = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
        
        elif kernel_type == "laplace_standard":
            kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        
        elif kernel_type == "laplace_diagonal":
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
        
        
        elif kernel_type == "sobel_horizontal":
            kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        elif kernel_type == "sobel_vertical":
            kernel = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
        
        elif kernel_type == "prewitt_horizontal":
            kernel = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        
        elif kernel_type == "prewitt_vertical":
            kernel = np.array([[-1, -1, -1],
                            [ 0,  0,  0],
                            [ 1,  1,  1]])
        
        elif kernel_type == "scharr_horizontal":
            kernel = np.array([[-3, -10, -3],
                            [0, 0, 0],
                            [3, 10, 3]])
        
        elif kernel_type == "scharr_vertical":
            kernel = np.array([[-3, 0, 3],
                            [-10, 0, 10],
                            [-3, 0, 3]])
        
        # Anwenden des Filters
        self.image = cv2.filter2D(self.image, -1, kernel)
    
    def binarize(self, method: str = "otsu", **kwargs) -> None:
        """
        Binarisiert ein OpenCV-Bild mit ausgewählten Schwellenwertmethoden.
        
        Args:
            method (str): Die ausgewählte Schwellenwertmethode.
                Mögliche Werte: "otsu", "global", "local_mean", "local_gaussian".
            **kwargs: Zusätzliche Parameter für die Schwellenwertmethoden.
                - Für die globale Methode:
                    threshold (int): Der Schwellenwert.
                - Für die lokale Methoden:
                    block_size (int): Größe des Blockes.
                    c (int): Konstante, die vom Mittelwert abgezogen wird.
        
        Returns:
            None
        """
        
        # Überprüfen, ob das Bild gültig ist
        if self.image is None:
            logging.error("Ungültiges Eingabebild. Stellen Sie sicher, dass ein gültiges Bild übergeben wird.")
            raise ValueError("Ungültiges Eingabebild. Stellen Sie sicher, dass ein gültiges Bild übergeben wird.")
        
        # Überprüfen, ob die ausgewählte Methode gültig ist
        valid_methods = ["otsu", "global", "local_mean", "local_gaussian"]
        if method not in valid_methods:
            logging.error(f"Ungültige Schwellenwertmethode '{method}' angegeben. Bitte wählen Sie zwischen {', '.join(valid_methods)}.")
            raise ValueError("Ungültige Schwellenwertmethode. Bitte wählen Sie zwischen 'otsu', 'global', 'local_mean' oder 'local_gaussian'.")
        
        # Anwenden der ausgewählten Methode
        if method == "otsu":
            self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
        elif method == "global":
            threshold = kwargs.get("threshold", 127)
            self.image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)[1]
            
        elif method == "local_mean":
            block_size = kwargs.get("block_size", 31)
            c = kwargs.get("c", 2)
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
            
        elif method == "local_gaussian":
            block_size = kwargs.get("block_size", 31)
            c = kwargs.get("c", 2)
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
            
    def histogram_equalization(self) -> None:
        """
        Führt einen Histogrammausgleich auf einem OpenCV-Bild in Graustufen durch.
        
        Returns:
            None
        """
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2,2))
        # Anwenden von CLAHE auf das Bild
        self.image = clahe.apply(self.image)
    
    def smoothen(self, filter_type: str = "bilateral") -> None:
        """
        Wendet einen Glättungsfilter auf ein OpenCV-Bild an.
        
        Args:
            filter_type (str): Der gewünschte Glättungsfilter.
                Mögliche Werte: "mean", "gauss", "median", "bilateral".
        
        Returns:
            None
        """
        
        # Überprüfen, ob das Bild gültig ist
        if self.image is None:
            logging.error("Ungültiges Eingabebild. Stellen Sie sicher, dass ein gültiges Bild übergeben wird.")
            raise ValueError("Ungültiges Eingabebild. Stellen Sie sicher, dass ein gültiges Bild übergeben wird.")
        
        # Überprüfen, ob der Filter-Typ gültig ist
        valid_filter_types = ["mean", "gauss", "median", "bilateral"]
        if filter_type not in valid_filter_types:
            logging.error("Ungültiger Filter-Typ. Bitte wählen Sie zwischen 'mean', 'gauss', 'median' oder 'bilateral'.")
            raise ValueError("Ungültiger Filter-Typ. Bitte wählen Sie zwischen 'mean', 'gauss', 'median' oder 'bilateral'.")
        
        # Anwenden des ausgewählten Filters
        if filter_type == "mean":
            self.image = cv2.blur(self.image, (3, 3))
            
        elif filter_type == "gauss":
            self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
            
        elif filter_type == "median":
            self.image = cv2.medianBlur(self.image, 3)
            
        elif filter_type == "bilateral":
            self.image = cv2.bilateralFilter(self.image, 9, 75, 75)
            
    def correct_skew(self) -> None:
        """
        Korrigiert die Verzerrung in einem gegebenen Bild.
        
        Returns:
            None
        """
        found_angle = round(determine_skew(self.image),2)
        
        logging.info(f"Erkannter Rotationswinkel: {-(found_angle)}")
        
        old_width, old_height = self.image.shape[:2]
        angle_radian = math.radians(found_angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, found_angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
                
        # Bild entzerren
        self.image = cv2.warpAffine(self.image, rot_mat, (int(round(height)), int(round(width))), borderValue=(255, 255, 255))
        
    
    def erode(self, kernel: np.ndarray, iterations: int = 1) -> None:
        """
        Führt eine Erosion auf einem OpenCV-Bild durch.
        
        Args:
            kernel (np.ndarray): Der Kernel für die Erosion.
            iterations (int): Die Anzahl der Iterationen für die Erosion (Standardwert ist 1).
        
        Returns:
            None
        """
        self.image = cv2.erode(self.image, kernel, iterations=iterations)

    def dilate(self, kernel: np.ndarray, iterations: int = 1) -> None:
        """
        Führt eine Dilatation auf einem OpenCV-Bild durch.
        
        Args:
            kernel (np.ndarray): Der Kernel für die Dilatation.
            iterations (int): Die Anzahl der Iterationen für die Dilatation (Standardwert ist 1).
        
        Returns:
            None
        """
        self.image = cv2.dilate(self.image, kernel, iterations=iterations)

    def opening(self, kernel: np.ndarray, iterations: int = 1) -> None:
        """
        Führt eine Öffnungsoperation auf einem OpenCV-Bild durch.
        
        Args:
            kernel (np.ndarray): Der Kernel für die Öffnungsoperation.
            iterations (int): Die Anzahl der Iterationen für die Öffnungsoperation (Standardwert ist 1).
        
        Returns:
            None
        """
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    def closing(self: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> None:
        """
        Führt eine Schließoperation auf einem OpenCV-Bild durch.
        
        Args:
            kernel (np.ndarray): Der Kernel für die Schließoperation.
            iterations (int): Die Anzahl der Iterationen für die Schließoperation (Standardwert ist 1).
        
        Returns:
            None
        """
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
    def thinning(self, iterations: int = 1) -> None:
        """
        Reduziert die Dicke der Zeichen (Buchstaben) auf eine Pixelbreite durch Anwendung der Thinning-Methode auf das Eingangsbild.

        Args:
            iterations (int): Die Anzahl der Iterationen für die Ausdünnung (Standardwert ist 1).

        Returns:
            None
        """
        # Invertiere das Bild, da die Thinning-Methode erwartet, dass die Objekte dunkel sind
        inverted_image = cv2.bitwise_not(self.image)
        
        # Anwenden der Thinning-Methode auf das invertierte Bild
        thinned_image = thin(inverted_image, max_num_iter=iterations)
        
        # Invertiere das Bild wieder, um die ursprüngliche Ausrichtung wiederherzustellen
        thinned_image = (thinned_image * 255).astype(np.uint8)
        self.image = cv2.bitwise_not(thinned_image)

    def get_image(self) -> np.ndarray:
        """
        Gibt das bearbeitete Bild zurück.

        Returns:
            Image.Image: Das bearbeitete Bild im PIL-Format.
        """
        return self.from_cv_to_pil(self.image)




class SymSpellSingleton:
    _instances = {}

    @classmethod
    def get_instance(cls, language: str):
        # Überprüfen, ob eine SymSpell-Instanz für die angegebene Sprache bereits existiert
        if language not in cls._instances:
            # Erstellen einer neuen SymSpell-Instanz, falls keine vorhanden ist
            cls._instances[language] = cls(language)
        # Rückgabe der SymSpell-Instanz für die angegebene Sprache
        return cls._instances[language]

    def __init__(self, language: str):
        # Initialisierung der SymSpell-Instanz mit den angegebenen Parametern
        self.language = language
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Laden des Wörterbuchs für die angegebene Sprache
        self.load_dictionary()

    def load_dictionary(self):
        # Definition der Wörterbuchdaten für verschiedene Sprachen
        language_data = {
            "en": {"path": "frequency_dictionary_en_82_765.txt", "url": None},
            "de": {"path": "de-100k.txt", "url": "https://github.com/wolfgarbe/SymSpell/blob/master/SymSpell.FrequencyDictionary/de-100k.txt"},
            "fr": {"path": "fr-100k.txt", "url": "https://github.com/wolfgarbe/SymSpell/blob/master/SymSpell.FrequencyDictionary/fr-100k.txt"},
            "es": {"path": "es-100k.txt", "url": "https://github.com/wolfgarbe/SymSpell/blob/master/SymSpell.FrequencyDictionary/es-100k.txt"}
        }
        # Abrufen der Wörterbuchinformationen für die angegebene Sprache
        lang_info = language_data.get(self.language.lower())
        if not lang_info:
            logging.info(f"Warnung: Sprache '{self.language}' wird nicht unterstützt. Der Spellcheck wird übersprungen.")
            return False
        
        # Bestimmen des Pfads zum Wörterbuch
        dictionary_path = pkg_resources.resource_filename("symspellpy", lang_info["path"])
        # Herunterladen des Wörterbuchs, falls es nicht vorhanden ist und eine URL angegeben ist
        if not os.path.exists(dictionary_path) and lang_info["url"]:
            urllib.request.urlretrieve(lang_info["url"], dictionary_path)
            logging.info("Wörterbuch von '{dictionary_path}' geladen.")
        # Laden des Wörterbuchs in die SymSpell-Instanz
        if not self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            logging.info("Fehler: Wörterbuch konnte nicht von '{dictionary_path}' geladen werden.")
            return False
        return True
    
    
class ResourceCache:
    _instance = None

    def __new__(cls):
        # Singleton-Pattern: Überprüfen, ob eine Instanz bereits existiert
        if cls._instance is None:
            # Erstellen einer neuen Instanz, wenn keine vorhanden ist
            cls._instance = super(ResourceCache, cls).__new__(cls)
            # Initialisieren der verschiedenen Ressourcen-Dictionaries
            cls._instance.stanza_pipelines = {}
            cls._instance.spell_checkers = {}
            cls._instance.sym_spells = {}
            cls._instance.stop_words = {}
            cls._instance.stemmers = {}
        return cls._instance

    def get_stanza_pipeline(self, language):
        # Überprüfen, ob die Stanza-Pipeline für die angegebene Sprache bereits vorhanden ist
        if language not in self._instance.stanza_pipelines:
            # Stanza für die angegebene Sprache herunterladen und Pipeline erstellen
            stanza.download(language, verbose=False)
            if language == 'multilingual':
                self._instance.stanza_pipelines[language] = stanza.Pipeline(lang='multilingual', processors='langid', langid_lang_subset=['de', 'en', 'fr', 'es', 'ja'], verbose=False)
            else:
                self._instance.stanza_pipelines[language] = stanza.Pipeline(lang=language, processors='tokenize,mwt,pos,lemma', use_gpu=False)
        # Rückgabe der Stanza-Pipeline für die angegebene Sprache
        return self._instance.stanza_pipelines[language]

    def get_spell_checker(self, language):
        # Überprüfen, ob der Rechtschreibprüfer für die angegebene Sprache bereits vorhanden ist
        if language not in self._instance.spell_checkers:
            # Initialisierung eines neuen PySpellChecker für die angegebene Sprache
            self._instance.spell_checkers[language] = PySpellChecker(language=language)
        # Rückgabe des Rechtschreibprüfers für die angegebene Sprache
        return self._instance.spell_checkers[language]

    def get_sym_spell(self, language):
        # Rückgabe der SymSpell-Instanz für die angegebene Sprache
        return SymSpellSingleton.get_instance(language).sym_spell

    def get_stop_words(self, language):
        # Überprüfen, ob die Stoppwörter für die angegebene Sprache bereits vorhanden sind
        if language not in self._instance.stop_words:
            # Herunterladen der NLTK-Stoppwörter für die angegebene Sprache
            nltk.download('stopwords', quiet=True)
            self._instance.stop_words[language] = set(stopwords.words(language))
        # Rückgabe der Stoppwörter für die angegebene Sprache
        return self._instance.stop_words[language]

    def get_stemmer(self, language):
        # Überprüfen, ob der Stemmer für die angegebene Sprache bereits vorhanden ist
        if language not in self._instance.stemmers:
            # Initialisierung eines neuen SnowballStemmer für die angegebene Sprache
            self._instance.stemmers[language] = SnowballStemmer(language)
        # Rückgabe des Stemmers für die angegebene Sprache
        return self._instance.stemmers[language]


class OCRPostProcessor:
    def __init__(self, text: str) -> None:
        """
        Initialisiert die OCRPostProcessor-Klasse mit dem übergebenen Text.
        
        Args:
            text (str): Der Eingabetext, der verarbeitet werden soll.
        """
        self.text: str = text
        self.language_mapping: dict = {"en": "english", "de": "german", "fr": "french", "es": "spanish"}
        self.resources: ResourceCache = ResourceCache()
        self.language: str = ""  # Sprachvariable initialisieren
    
    def identify_language(self) -> None:
        """
        Identifiziert die Sprache der Tokens mithilfe der Stanza-Pipeline.
        
        Returns:
            None
        """
        stanza.download(lang="multilingual", verbose=False)
        # Initialisieren des Stanza-Pipeline mit der 'langid' Komponente
        nlp = self.resources.get_stanza_pipeline('multilingual')
        
        # Anwenden der Pipeline auf den Text
        doc = nlp(self.text)
        
        # Extrahieren der erkannten Sprache
        self.language = doc.lang
        logging.info("Folgende Sprache wurde erkannt: {self.language}")
    
    def correct_line_breaks(self) -> None:
        """
        Korrigiert Zeilenumbrüche im Text.
        
        Returns:
            None
        """
        # Entfernen von Zeilenumbrüchen, die mit einem Bindestrich verbunden sind, und Ersetzen durch Leerzeichen
        self.text = self.text.replace('-\n', '').replace('\n', ' ')
        
    def lowercase(self) -> None:
        """
        Konvertiert den Text in Kleinbuchstaben.
        
        Returns:
            None
        """
        self.text = self.text.lower()

    def remove_special_characters(self) -> None:
        """
        Entfernt Sonderzeichen aus dem Text.
        
        Returns:
            None
        """     
        # Ersetzen von Sonderzeichen durch Leerzeichen
        self.text = re.sub(r'\W', ' ', self.text)
        
    def remove_extra_spaces(self) -> None:
        """
        Entfernt zusätzliche Leerzeichen aus dem Text.
        
        Returns:
            None
        """
        
        # Entfernen von Leerzeichen am Anfang und Ende des Textes
        self.text = self.text.strip()
        
        # Ersetzen von mehreren aufeinanderfolgenden Leerzeichen durch ein Leerzeichen
        self.text = ' '.join(self.text.split())

    def remove_stopwords(self) -> None:
        """
        Entfernt Stoppwörter basierend auf der erkannten Sprache.
        
        Returns:
            None
        """
        # Überprüfen, ob die Sprache unterstützt wird
        if self.language.lower() in self.language_mapping:
            stop_words = self.resources.get_stop_words(self.language_mapping.get(self.language, 'english'))
            
            # Text in einzelne Wörter aufteilen
            words = self.text.split()
            
            # Entfernen von Stoppwörtern
            words = [word for word in words if word.lower() not in stop_words]
            
            # Verbinden der Wörter zu einem Text
            self.text = ' '.join(words)
        else:
            logging.info(f"Warnung: Sprache '{self.language}' wird nicht unterstützt. Das Entfernen von Stoppwörtern wird für diese Sprache übersprungen.")

    def lemmatize(self) -> None:
        """
        Führt eine Lemmatisierung des Textes durch.
        
        Returns:
            None
        """
        # Überprüfen, ob die erkannte Sprache unterstützt wird
        if self.language.lower() in self.language_mapping:
            # Abrufen der Stanza-Pipeline für die erkannte Sprache
            nlp = self.resources.get_stanza_pipeline(self.language)
            # Anwenden der Pipeline auf den Text
            doc = nlp(self.text)
            # Extrahieren und Lemmatisieren der Tokens
            self.text = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
        else:
            logging.info(f"Warnung: Sprache '{self.language}' wird nicht unterstützt. Die Lemmatisierung wird für diese Sprache übersprungen.")

    def stem(self) -> None:
        """
        Wendet Stemming auf den Text an.
        
        Returns:
            None
        """
        # Überprüfen, ob die erkannte Sprache unterstützt wird:
        if self.language.lower() in self.language_mapping:
            stemmer = self.resources.get_stemmer(self.language_mapping.get(self.language, 'english'))
            # Tokenisierung des Textes
            word_tokens = word_tokenize(self.text)
                
            # Stemming für die aktuelle Sprache durchführen
            stemmed_words = [stemmer.stem(word) for word in word_tokens]
                
            self.text = ' '.join(stemmed_words)
        else:
            logging.info(f"Warnung: Sprache '{self.language}' wird nicht unterstützt. Das Stemming wird übersprungen.")

    def spellcheck(self, checker_type: str = "symspell") -> None:
        """
        Führt eine Rechtschreibprüfung durch, je nach gewähltem Typ (SymSpell oder PySpellChecker).
        
        Args:
            checker_type (str): Der Typ des Rechtschreibprüfers. Standardmäßig ist "symspell".
                                Möglicheu Auswah. "symspell", "pyspell"
        
        Returns:
            None
        """
        # Überprüfen, ob die erkannte Sprache unterstützt wird
        if self.language.lower() in self.language_mapping:
            if checker_type == "symspell":
                # Laden des SymSpell-Rechtschreibprüfers
                sym_spell = self.resources.get_sym_spell(self.language)
                # Aufteilen des Textes in Wörter
                words = self.text.split()
                # Korrigieren der Wörter mit SymSpell
                corrected_words = [suggestions[0].term if (suggestions := sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)) else word for word in words]
                # Verbinden der korrigierten Wörter zu einem Text
                self.text = ' '.join(corrected_words)
            elif checker_type == "pyspell":
                # Laden des PySpellChecker-Rechtschreibprüfers
                spell_checker = self.resources.get_spell_checker(self.language_mapping.get(self.language, 'english'))
                # Aufteilen des Textes in Wörter
                words = self.text.split()
                # Korrigieren der Wörter mit PySpellChecker
                corrected_words = [corrected_word if (corrected_word := spell_checker.correction(word)) else word for word in words]
                # Verbinden der korrigierten Wörter zu einem Text
                self.text = ' '.join(corrected_words)
            else:
                logging.info(f"{checker_type} wird nicht unterstützt. Nutzen Sie 'symspell' oder 'pyspell'. Die Rechtschreibprüfung wird übersprungen.")
        else:
            logging.info(f"Warnung: Sprache '{self.language}' wird nicht unterstützt. Die Rechtschreibprüfung wird übersprungen.")

    def get_text(self) -> str:
        """
        Gibt den verarbeiteten Text zurück.
        
        Returns:
            str: Der verarbeitete Text.
        """
        return self.text