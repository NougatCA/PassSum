# PassSum

## Requirements

### Minimize requirements

The list of minimize requirements can be found in `requirements.txt`.

### Additional requirements

If you need to reprocess the raw dataset, or use your own dataset,
then you will also need to install the following packages.
```
tree_sitter==0.19.0
```

If you encounter errors about `my-languages.so` when preprocessing the dataset, 
please run `sources/data/asts/build_lib.py` first.

## Datasets and Tokenizers

We provide pre-processed datasets, saved as pickle binary files, 
which can be loaded directly as instances.

The pre-processed datasets can be downloaded [here](https://1drv.ms/u/s!Aj4XBdlu8BS0geofs21I6RwjMY_jxg?e=IVpDhG).
Put the downloaded dataset pickle file into `{dataset_root}/dataset_saved/` (default to`.../dataset/dataset_saved`), 
the program will automatically detect and use it.

It is also possible to use a custom dataset, 
simply by placing it in the specified location according to the relevant settings in the source code, 
or by modifying the corresponding dataset loading script in the source code. 
The dataset loading code is located in the `sources/data/data.py` and `sources/data/data_utils.py` files.

##  Pre-trained Tokenizers and Models

Custom tokenizers (we call "vocab") can be downloaded [here](https://1drv.ms/u/s!Aj4XBdlu8BS0geogAjghvdjk2TRzuA?e=hXNekW). Extract it in a certain directory. 
Specific the argument `trained_vocab` of `main.py` 
where the tokenizers are located or put it in `{dataset_root}/vocab_saved` (default to`.../dataset/vocab_saved`).

Pre-trained models available [here](https://1drv.ms/u/s!Aj4XBdlu8BS0geoh2I7bt_HtLGA4yA?e=4PPrK7).
Extract and put it in a directory, then specific the argument `trained_model` like tokenizers before.

## Runs

Run `main.py` to start running. 
All arguments are located in `args.py`, specific whatever you need.

Example scripts is as following.
```shell
python main.py \
--batch-size 64 \
--eval-batch-size 128 \
--n-epoch 200 \
--cuda-visible-devices 0,1 \
--summarization-language java \
--model-name summarization_java
```