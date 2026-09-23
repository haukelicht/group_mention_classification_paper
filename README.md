
# Replication materials for "Attributes, Abstractions, and Combinations: A New Taxonomy for Studying How Parties Invoke Social Groups"

This repository contains the replication materials for the paper "Attributes, Abstractions, and Combinations: A New Taxonomy for Studying How Parties Invoke Social Groups" by [Hauke Licht](https://haukelicht.github.io/) and [Leonce Röth](https://scholar.google.com/citations?user=IEV3CNoAAAAJ):

```bibtex
@article{licht_roeth_2026,
  title={Attributes, Abstractions, and Combinations: A New Taxonomy for Studying How Parties Invoke Social Groups},
  author={Hauke Licht and Leonce Röth},
  journal={Journal of European Public Policy},
  volume={},
  number={},
  pages={},
  year={2026},
  date={2026-09},
  url={http://www.tandfonline.com/10.1080/13501763.2026.2737691},
  doi={10.1080/13501763.2026.2737691}
}
```

Pre-trained classification models in the Hugging Face 🤗 model hub: [_social group mention attribute classification_ collection](https://huggingface.co/collections/haukelicht/social-group-mention-attribute-classification)

Access the **pre-print** on OSF: <https://osf.io/g7htx/files/3qdzc>

Corresponding author: Hauke Licht (hauke.licht@uibk.ac.at)

## Repository structure

The repository is organized as follows:

- `code/`: Contains all scripts for data classification and analyses.
- `data/`: Contains the raw and processed data used in the analyses.
- `src/`: Contains the Python source code modules for the project.
- `results/`: Contains the output of the analyses, including tables, figures, and other intermediate result files.
- `manuscript/`: Contains the manuscript and supplementary materials (TeX source and PDF files).
- `requirements.txt`: Lists all Python dependencies required for the project.
- `README.md`: This file, providing an overview and instructions for reproducing the analyses.

## Replication

### Reproducing the analyses

`code/run_analyses.py` reproduces the analyses reported in the paper and supporting materials and creates all necessary output and intermediate files.

_Note:_ The reproducibility run _excludes_ preparing the case selection, manifesto text data, and  classifier training and inference.

#### Python setup {#python-setup}

All python dependencies are listed in the [`requirements.txt`](requirements.txt) file.

I have used the following commands to set up a Python environment using conda:

```bash
# create the conda environment for the project
conda create -n group_mention_classification python=3.11 pip -y
# activate it
conda activate group_mention_classification
# install the required Python dependencies in it
pip install -r requirements.txt
```

#### Executing the reproducibility run

```bash
# ensure that the root folder of the repository is included in the Python path so modules in src/ can be imported
export PYTHONPATH="$(pwd):$PYTHONPATH" 
# activate the conda environment for the project
conda activate group_mention_classification
# navigate to the code directory
cd code
# run the analyses script
python run_analyses.py
```

### Producing the classifier inference results

It is not necessary to re-run the classifier inference to reproduce the reported results because all inference outputs necessary to run the analyses are cached in `data/labeled` and the resulting main data files already exists as `data/labeled_mentions_with_party_metadata.pkl`.

However, if you wish to run the classifier inference (e.g. with your own data), use the appropriate `run_inference*.sh` scripts in the `code/classification/` subfolders:

- `code/classification/group_mention_detection/`
- `code/classification/group_mention_classification/`

_Note:_ These scripts assume that 

1. you have the conda environment named `group_mention_classification` set up and installed all required dependencies in it (see the [Python setup](#python-setup) section above)
2. your `PYTHONPATH` contains the `src/` directory of this repository, which contains the inference modules in `src/inference/`.

The same applies to running the `finetuning_*.sh` scripts in `code/classification/group_mention_classification/finetuning/` for fine-tuning the group attribute multi-label classifier models on our labeled data.
