# ProGen2 Finetuning  🦾 🧬 🧪

*Accompanying code for my bachelor thesis and [paper](https://doi.org/10.1109/BIBM62325.2024.10821712).*

**Ever wanted to finetune a generative protein language model on protein families of your choice? No? Well, now you can!**

## Usage

We describe a simple workflow, in which we finetune the [ProGen2-small](https://github.com/salesforce/progen/tree/main/progen2) (`151M`) model illustrate the usage of the provided python scripts.

### Install dependencies

First of all, we need to install the required dependencies. Use a virtual environment to avoid conflicts with the system-wide packages.

```bash
cd src
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Downloading data

Select a few families from the [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/#table) database, which you want to train the model on. Use their Pfam codes to download the data in FASTA format. The downloaded files will be saved into the `downloads/` directory. This may take a while, depending on the size of the downloaded families.

Example code to dowlnoad three, relatively small, protein families:

```bash
python3 download_pfam.py PF00257 PF02680 PF12365 
```

### Preprocessing the data

Before finetuning the model, we need to preprocess the data to include the special famliy tokens, and the `1` and `2` tokens at the beginning and end of sequence. We also remove the FASTA headers.

We specify the paths to the downloaded FASTA files using the `--input_files` option.  
Optionally, we may define the names of output train and test data files in which the data will be stored. We can also specify the ratio of train and test data split (default is 0.8) and using a boolean flag `--bidirectional` we can save the sequences also in reverse, if we want to train a bidirectional model.

```bash
python3 prepare_data.py \
    --input_files downloads/PF00257.fasta downloads/PF02680.fasta downloads/PF12365.fasta \
    --output_file_train=train_data_3_fams.txt \
    --output_file_test=test_data_3_fams.txt \
    --train_split_ratio=0.8 \
    --bidirectional
```

### Finetuning

Now we can finetune the model on the prepared data. It is highly recommended to use a GPU for finetuning. We specify paths to the train and test files and the values of hyperparameters. The base model weights are automatically downloaded from my huggingface [repo](https://huggingface.co/hugohrban/progen2-small). After finetuning, the model binary, config file and tokenizer are saved into the `checkpoints/` directory.

```bash
python3 finetune.py \
    --model=hugohrban/progen2-small \
    --train_file=train_data_3_fams.txt \
    --test_file=test_data_3_fams.txt \
    --device=cuda \
    --epochs=5 \
    --batch_size=16 \
    --accumulation_steps=4 \
    --lr=1e-4 \
    --decay=cosine \
    --warmup_steps=200 \
    --eval_before_train
```

Run `python3 finetune.py --help` to see the full list of available options and their descriptions.

### Sampling

Use the `sample.py` script to generate new sequences from a model using top-k sampling with temperature. You may use the model finetuned on 7 families described in the thesis, which is also available in my huggingface as [progen2-small-mix7](https://huggingface.co/hugohrban/progen2-small-mix7), or its bidirectional version [progen2-small-mix7-bidi](https://huggingface.co/hugohrban/progen2-small-mix7-bidi).

```bash
python3 sample.py \
    --model=hugohrban/progen2-small-mix7 \
    --device=cuda \
    --batch_size=8 \
    --iters=1 \
    --max_length=512 \
    --t=1.0 \
    --k=10 \
    --prompt="<|pf03668|>1MEVVIVTGMSGAGK"
```

Use the `--help` or `-h` option to see the full list of available options and their descriptions.

## Loading the model directly

You can load the model directly from the huggingface repository and use it in python. All of the models are registered for the `AutoModelForCausalLM` class, so you don't even need to have the model source code and config files stored locally. The following code snippet shows how to use the model to predict the next token probabilities given a prompt.

```python
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
# optionally use local imports
# from models.progen.modeling_progen import ProGenForCausalLM
# from models.progen.configuration_progen import ProGenConfig
import torch

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-small-mix7", trust_remote_code=True)
tokenizer = Tokenizer.from_pretrained("hugohrban/progen2-small-mix7")
tokenizer.no_padding()

# prepare input
prompt = "<|pf03668|>1MEVVIVTGMSGAGK"
input_ids = torch.tensor(tokenizer.encode(prompt).ids).to(model.device)

# forward pass
logits = model(input_ids).logits

# print next token probabilities
next_token_logits = logits[-1, :]
next_token_probs = torch.softmax(next_token_logits, dim=-1)
for i in range(tokenizer.get_vocab_size(with_added_tokens=False)):
    print(f"{tokenizer.id_to_token(i)}: {round(100 * next_token_probs[i].item(), 2):.2f} %")
```

## Citation

If you found this code useful, please cite:
```bibtex
@INPROCEEDINGS{10821712,
  author={Hrbáň, Hugo and Hoksza, David},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Protein Family Sequence Generation through ProGen2 Fine-Tuning}, 
  year={2024},
  volume={},
  number={},
  pages={7037-7039},
  keywords={Proteins;Training;Measurement;Adaptation models;Biological system modeling;Molecular biophysics;Protein sequence;Natural language processing;Biological information theory;Software development management;protein sequence;protein language model;sequence generation;fine-tuning},
  doi={10.1109/BIBM62325.2024.10821712}}
```
