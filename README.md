### PRISM 

This is the code repository for the paper [On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective ](https://arxiv.org/abs/2505.11318) by Donald Loveland, Mingxuan Ju, Tong Zhao, Neil Shah, and Danai Koutra. The paper explores the impact of weight decay on collaborative filtering models, particularly in the context of popularity bias in angle-based recommendation systems.

--- 

## Requirements

PRISM simply requires setting the magnitude of the embedding vectors during initialization based on the popularity of the users/items. While this only requires base PyTorch, this is typically supported by more complex dataloading frameworks. Specifically, we use PyTorch Geometric to load the data. The requirements are provided in the `environment.yml` file. You can install them using:

`conda env create -f environment.yml`

## Preparing Data

To run the code, you will need to prepare the data. We provide a script `dataloader.py` that loads and splits the data into training, validation, and test sets. The script uses the MovieLens 1M dataset as an example, but you can modify it to use other datasets as needed. For models that use message passing, such as LightGCN, you will need to set the `num_layers` argument in the script to ensure the dataloader samples the appropriate neighborhoods. This can be done with 
`python dataloader.py --num_layers K`. 

## Training a Model

To train a model, you can use the provided scripts `./run_train_prism.sh`. The scripts are designed to work with the MovieLens 1M dataset that was prepped within the dataloader, but you can modify them to work with other datasets as needed. 

Current parameters to include within the script, with their default values, are:

DATASET='MovieLens1M' - Dataset to use 
MODEL='MLP' - Model backbone 
EPOCHS=1000 - Number of epochs to train the model
LOSS='align' - Loss function to train with 
REG='uniformity' - Regularization to use 
GAMMA='1.0' - Regularization strength
WD='0.0' - Weight decay strength
LR='0.001' - Learning rate
PATIENCE='10' - Early stopping patience
DEGREE_INIT_STR='1.0' - PRISM initialization strength
DATA_PATH='model_chkps' - Path to save model checkpoints

## Running Tests
To run tests on the trained model, you can use the provided script `./test.sh`. This script will evaluate the model on the MovieLens1M test set and output the results. The default parameters are set in the script, but again, you can modify them as needed.

## Building on Code

1.  `losses.py` - Contains the loss functions and regularizations used in the paper. Any new losses or regularizations should be added here.
2.  `models.py` - Contains the model architectures used in the paper. Any new models should be added here.
3.  `dataloader.py` - Contains the data loading and preprocessing logic. If you want to use a different dataset, you can modify this file accordingly.


## Citation

If you find this work useful, please cite it as:

```bibtex
@inproceedings{loveland2025weightdecay,
  author       = {Loveland, Donald and Ju, Mingxuan and Zhao, Tong and Shah, Neil and Koutra, Danai},
  title        = {On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective},
  booktitle    = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)},
  year         = {2025},
  doi          = {10.1145/3711896.3737068},
  url          = {https://doi.org/10.1145/3711896.3737068}
}
