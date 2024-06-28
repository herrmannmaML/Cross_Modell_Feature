FROM python:3.11.7

RUN apt-get -y update && apt-get -y install git libgl1 gcc g++ && pip install poetry==1.7.1
RUN apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn -y

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONPATH=${PYTHONPATH}:${PWD}

WORKDIR /code

COPY pyproject.toml ./

RUN poetry install --no-root --no-directory