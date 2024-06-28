import warnings
import logging
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

# Logging-Config
logging.basicConfig(filename='../project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Zeige keine Warnungen an
warnings.filterwarnings("ignore")

class DataLoader:
    """
    Eine Klasse zur Verarbeitung von HuggingFace-Datensätzen.
    
    Args:
        dataset_name (str): Der Name des HuggingFace-Datensatzes, der heruntergeladen werden soll.
        save_path (str): Der lokale Pfad, an dem der Datensatz gespeichert werden soll.
    """

    def __init__(self, dataset_name: str, save_path: str):
     ‚   self.dataset_name = dataset_name
        self.save_path = save_path

    def load_and_preprocess(self):
        """
        Lädt den Datensatz herunter und vorverarbeitet ihn.
        """
        self.dataset = load_dataset(self.dataset_name, cache_dir=self.save_path)
        logging.info(f"Datensatz {self.dataset_name} erfolgreich heruntergeladen")
        
        # Features des Datensatzes ausgeben
        logging.info(f"Features im Datensatz: {self.dataset[next(iter(self.dataset.keys()))].features}")
        
        # Labels zu doc_category konvertieren
        label_names = self.dataset[next(iter(self.dataset.keys()))].features['label'].names
        self.dataset = self.dataset.map(lambda x: {'doc_category': label_names[x['label']]})
        
        # Bilder von Grauwert zu RGB konvertieren
        #self.dataset = self.dataset.map(lambda x: {'image': x['image'].convert('RGB') if 'image' in x else x})
        
        # Dataset splitten
        self.dataset = self.split_dataset()
        
        # Das DatasetDict speichern
        self.dataset.save_to_disk(self.save_path)
        logging.info(f"Datensatz '{self.dataset_name}' unter '{self.save_path}' gespeichert!")
        
    def split_dataset(self):
        """
        Splittet den Datensatz in train, validation und test und gibt ein DatasetDict zurück.
        """
        def stratified_split(dataset, test_size=0.15, val_size=0.15):
            labels = np.array(dataset['label'])
            train_val_indices, test_indices = train_test_split(
                np.arange(len(labels)), test_size=test_size, stratify=labels
            )
            train_val_labels = labels[train_val_indices]
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=val_size / (1 - test_size), stratify=train_val_labels
            )
            return train_indices, val_indices, test_indices

        train_indices, val_indices, test_indices = stratified_split(self.dataset['train'])
        train_dataset = self.dataset['train'].select(train_indices)
        val_dataset = self.dataset['train'].select(val_indices)
        test_dataset = self.dataset['train'].select(test_indices)
        
        # Entfernen des Features 'image'
        train_dataset = train_dataset.remove_columns(['label'])
        val_dataset = val_dataset.remove_columns(['label'])
        test_dataset = test_dataset.remove_columns(['label'])
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })