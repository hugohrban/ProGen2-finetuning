# ProGen2 Finetuning

Accompanying code for my bachelor thesis.

### Workflow
We describe a simple workflow in which we illustrate the usage of the indifidual python scripts.

First of all, `cd src` into the folder which contains the source code.

#### Downloading data
Select a few families from tha Pfam database, which you want to train the model on. Use their Pfam codes to download the data in FASTA format. The downloaded files will be saved into the `downloads/` directory. This may take a while, depending on the size of the downloaded families.

```bash
python3 download_pfam.py PF12300 PF12365 PF00257
```

This command will download three protein families.


#### Preprocessing the data
Before finetuning the model, we need to preprocess the data to include the special famliy tokens, and the "1" and "2" tokens. We also remove the FASTA headers.

We specify the paths to the downloaded FASTA files using the `--input_files` option. 
Optionally, we may define the names of train and test data files in which the data will be stored. We can also specify the ratio of train and test data (default is 0.8) and using a boolean flag `--bidirectional` we can save the sequences also in reverse, if we want to train a bidirectional model.

```bash
python3 prepare_data.py \
    --input_files downloads/PF12300.fasta downloads/PF12365.fasta downloads/PF00257.fasta \
    --output_file_train=train_data_3fam.txt \
    --output_file_test=test_data_3fam.txt \
    --train_split_ratio=0.8 \
    --bidirectional
```


#### Finetuning





