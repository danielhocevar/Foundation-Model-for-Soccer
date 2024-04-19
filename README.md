# A Foundation Model for Soccer
##### By: Ethan Baron, Daniel Hocevar and Zach Salehe

We propose a foundation model for soccer, which is able to predict subsequent actions in a soccer match from a given input sequence of actions. As a proof of concept, we train a transformer architecture on three seasons of data from a professional soccer league. We quantitatively and qualitatively compare the performance of this transformer architecture to a baseline Markov model, as well as an MLP model. We discuss potential applications of our model and associated ethical considerations.

## Getting Started
We provide a requirements.txt file which contains the required python packages to run our project. The packages can be installed with the following command
```
pip install -r requirements.txt
```

## Navigating the Codebase

#### Training a model
- The notebook `train.ipynb` showcases how a training dataset can be loaded and a model can be trained from scratch.

#### Evaluating a model
- The notebook `eval.ipynb` is used to construct the table in our report which compares the performance of each of the models

#### Viewing Model Definitions
- The `/models` directory contains definitions for each of the neural networks we use. 

#### Investigating scaling laws
- The notebook `scaling_laws.ipynb` contains several tests exploring how changing the size of the dataset or the number of parameters in the model affects the model's validation accuracy

#### Accessing pretrained weights
- The weights for the models we have trained can be found in the `/pretrained` folder. We provide helper functions in `/pretrain/load_pretrained.py` to load both the small and large variations of our transformer model.

#### Preprocessing data
The notebook `download_data.ipynb` and the file `preprocess_data.py` contain code for downloading the dataset, and preprocessing it into the format required by the models.

#### Visualizing model embeddings
The notebook `embeddings_viz.ipynb` contains code for extracting embeddings from the model and visualizing them.
