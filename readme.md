# The UEA Digital Humans Entry to the GENEA Challenge 2023

This repository contains implementations for  [The UEA Digital Humans Entry to the GENEA Challenge 2023](https://openreview.net/pdf?id=bBrebR1YpXe). This is a self and cross-attention adaptation of the [Transformer-XL](https://github.com/kimiyoung/transformer-xl) model.



## Data Pre-Processing

This part of the repository is a work in progress and will be updated with an easier-to-run script to extract features in future. 

In the meantime for each GENEA data split, i.e. train, validation and test, please edit the `data_root` variable (L157) in the following file to the root folder of the GENEA data split, for example, `genea_data/train` and run:

```
python pre-processing/genea_parser.py
```

To extract PASE+ features using the script provided, please download the pre-trained model and install the requirements for [PASE+](https://github.com/santi-pdp/pase) and place each file in the following directory:
```
pre-processing/pase_files/PASE+.cfg
pre-processing/pase_files/FE_e199.ckpt
```
Then for each GENEA data split, please edit the `split` variable (L20) in the following file and run for each train, validation and test split:

```
python pre-processing/pase_parser.py
```

Please download the FastText model and requirements from https://fasttext.cc/docs/en/english-vectors.html and place them in the `pre-processing/fasttext_files/crawl-300d-2M-subword.bin`` and run:

```
python pre-processing/generate_fasttext.py --tsv_dir genea_data/train/tsv --dest_dir data/tsv_fasttext --tensor_dir data/
```

The repo relies on data being pre-processed for each subset of train, validation and test stored in a file hierarchy as follows (omitting dof6 and pos for test data):

    .
    └── train
        ├── main-agent
        │   ├── pase_features
        │   ├── dof6
        │   ├── pos
        │   ├── fasttext
        │   └── audio
        └── interloctr
            ├── pase_features
            ├── dof6
            ├── pos
            ├── fasttext
            └── audio

Each subdirectory contains a numpy mmap file for each file in the dataset that can be shaped into ($n$, $f$) where $n$ is the number of motion frames and $f$ is the feature size for each frame.

While this repo uses PASE+ audio features and FastText text features, the repository can be used with other features by following the same directory format, however, change the respective feature size in the config file.

We have also provided the standard scalar used in training which was calculated using only the training data for the GENEA challenge 2023. 

## Config

This repository allows tweaking hyperparameters through the use of a config file used during training and inference.
We provide the config file used to generate the model used in our paper in ```configs/genea.yaml```.

## Training

To train our paper model from scratch, please prepare the data and run:
```
python train.py --config configs/genea.yaml
```

## Inference

We provide an inference script to generate the BVH files for a given test set. To infer all files in the processed test directory, run:

```
python inference.py --config <config path> --epoch <epoch number> --mem_len <memory length>
```
`--memory length` is optional and will default to the value provided in the config file if not provided.

The model used to generate the submitted BVH files will be provided at a later date.
