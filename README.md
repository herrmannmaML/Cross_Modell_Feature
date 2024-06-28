# Repository für Masterarbeit: Dokumentenklassifikation


## Projektbeschreibung

Die Masterarbeit für die DBUAS widmet sich dem Vergleich unterschiedlicher Deep-Learning-Methoden zur Dokumentenklassifikation anhand eines Open-Source-Datensatzes. Der praktische Teil der Arbeit, welcher dieses Code-Projekt darstellt, vergleicht verschiedene State-of-the-Art Deep Learning Methoden zur Dokumentenklassifikation mithilfe des Open-Source-Datensatzes.

</br>
Datensatz: "LINK_EINFÜGEN"
</br>
</br>

## :white_check_mark: Anforderungen
* Python 3.11.7
* Docker

</br>
</br>

## :file_folder: Repository Übersicht
  ```bash
├── data		       # enthält Daten
│
├── env
│   ├── Dockerfile                 # Erstellung der Entwicklungsumgebung
│
├── notebooks                  # Jupyter Notebooks
│
├── references                 # Data-Dictionaries, Dokumentation
│
├── reports                    # Berichte im Format HTML, PDF, LATEX usw.
│   ├── figures                     # erstellte Abbildungen/Grafiken für Berichte
│
├── src                        # Source-Code
│   ├── utils.py		    # Hilfsfunktionen für Projekt
│
├── .gitignore                 # Liste von Dateien/Verzeichnisen, die von Git ignoriert werden sollen
│
├── poetry.lock                # Dependency-Verwaltung von Poetry
│
├── .pre-commit-config.yaml    # Konfigurationsdatei für Pre-Commit-Hooks
│
├── pyproject.toml             # Konfigurationsdatei für Poetry und Pre-Commit-Hooks
│
├── README.md                  # Projektbeschreibung/Installations- und Nutzungsanweisungen
  ```

</br>
</br>

## :electric_plug: Installation

1. Wechsel zu Projektordner
   ```sh
   cd Documents\Projekte
   ```

2. Clone das Repository
   ```sh
   git https://github.com/herrmannmaML/Masterarbeit_DocClassification.git
   cd Masterarbeit_DocClassification
   ```

3. Erstelle die lokale Projektumgebung
   ```sh
   docker build -f env/Dockerfile -t cross_modal_feature_env . 
   ```

</br>
</br>

## :zap: Nutzung

Docker-Container starten für Entwicklung:
```sh
docker run -it --name cross_modal_feature_env -v $(pwd):/code cross_modal_feature_env
```


Erstelle das Docker-Compose-Cluster für Modell-Tracking + Entwicklung
```sh
docker-compose --env-file secrets/.env up -d --build
```