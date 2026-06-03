#!/usr/env/bin bash

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
