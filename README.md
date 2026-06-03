# <ins>F</ins>orward-<ins>L</ins>ooking <ins>A</ins>ctive <ins>RE</ins>trieval augmented generation (FLARE)

This repository contains the code and data for the paper
[Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983).

## Overview

FLARE is a generic retrieval-augmented generation method that actively decides when and what to retrieve using a prediction of the upcoming sentence to anticipate future content and utilize it as the query to retrieve relevant documents if it contains low-confidence tokens.

<p align="center">
  <img align="middle" src="res/flare.gif" height="350" alt="FLARE"/>
</p>

## Install environment

```shell
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

Or simply run:
```shell
./setup.sh
```

## Prerequisites

### Ollama Setup
This project uses [Ollama](https://ollama.ai/) for LLM inference. Install Ollama and pull the model:

```shell
# Install Ollama (see https://ollama.ai/download)
ollama pull qwen3:8b
```

Configure via environment variables:
```shell
export OLLAMA_MODEL=qwen3:8b           # Model to use (default: qwen3:8b)
export OLLAMA_BASE_URL=http://localhost:11434  # Ollama server URL (default)
```

## Quick start

### Download Wikipedia dump
Download the Wikipedia dump from [the DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:
```shell
mkdir data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

### Build local vector index
We use FAISS with sentence-transformers embeddings for local retrieval (replacing Elasticsearch/Bing):
```shell
python prep.py --task build_index --inp data/dpr/psgs_w100.tsv wikipedia_dpr
```

You can also use ChromaDB as the backend:
```shell
python prep.py --task build_index --inp data/dpr/psgs_w100.tsv wikipedia_dpr --engine chromadb
```

### Run FLARE
Use the following command to run FLARE with a locally hosted Ollama model:
```shell
./run.sh 2wikihop configs/2wikihop_flare_config.json  # 2WikiMultihopQA dataset
./run.sh wikiasp configs/wikiasp_flare_config.json  # WikiAsp dataset
```

Set `debug=true` in `run.sh` to activate the debugging mode which walks you through the iterative retrieval and generation process one example at a time.

## Architecture

- **LLM Backend**: Ollama (locally hosted, default model: `qwen3:8b`)
- **Retrieval**: FAISS or ChromaDB with sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- **NER**: Hugging Face transformers pipeline (`dslim/bert-base-NER`)
- **Sentence Tokenization**: NLTK Punkt

## Citation
```
@article{jiang2023flare,
      title={Active Retrieval Augmented Generation}, 
      author={Zhengbao Jiang and Frank F. Xu and Luyu Gao and Zhiqing Sun and Qian Liu and Jane Dwivedi-Yu and Yiming Yang and Jamie Callan and Graham Neubig},
      year={2023},
      eprint={2305.06983},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
