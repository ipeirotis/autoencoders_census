Step 1: data_loader, loads the dataset and returns back a processed dataframe that can then be encoded/decoded

Step 2: autoencoder, sets up the architecture of the autoencoder

Step 3: hyperparameter_tuning, finds the optimal parameters for the neural network for the autoencoder

Step 4: measure_accuracy, measures how we well we can reconstruct each attribute of the dataset

Step 5: fine_outliers, examines which rows of the dataset are the most difficult to reconstruct (and therefore are likely to come from inattentive/mischevious participants)


## How to run

The main file is `main.py`. You can run various entrypoints by choosing the appropriate arguments.
The available entrypoints are the following: 

### Train model
```bash
python main.py train 
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to train. Choose between two available values: `AE` for the simple autoencoder and `VAE` for a variational autoencoder.
- `--prior`: Prior to use for the variational autoencoder. Choose between two available values: `gaussian` for a Gaussian prior and `gumbel` for applying the gumbel softmax.
- `--data`: Dataset to train the model on. Default: `sadc_2017`.
- `--config`: Configuration file for the model training. This should contain the hyperparameters for the model. You can find an example of the configuration file in the `config` folder and the files that have `simple` as a prefix.
- `--output`: Output folder to save the outputs. Default: `cache/simple_model/`.

The output of this command is a trained model that can be used for evaluation and is stored in the output folder under the `autoencoder` subfolder.
Also, it generated a `.npy` file of the history as produced by the keras fit method for future use, as well as plots of training and validation losses; 
Specifically, for `AE` it stores the reconstruction loss plot, whereas for `VAE` the reconstruction loss, the kl loss, as well as the total cost of these two.


### Search Hyperparameters
```bash
python main.py search_hyperparameters
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to search hyperparameters for. Choose between two available values: `AE` for the simple autoencoder and `VAE` for a variational autoencoder.
- `--prior`: Prior to use for the variational autoencoder. Choose between two available values: `gaussian` for a Gaussian prior and `gumbel` for applying the gumbel softmax.
- `--data`: Dataset to train the model on and search for hyperparameters. Default: `sadc_2017`.
- `--config`: Configuration file for the model hyperparameters searching. This should contain the hyperparameters for the model. You can find an example of the configuration file in the `config` folder and the files that have `hp` as a prefix.
- `--output`: Output folder to save the outputs. Default: `cache/simple_model/`.

The output of this command is a `yaml` file that contains the best hyperparameters found during the search. This file is stored in the output folder in a file named as `best_hyperparameters.yaml`.

### Evaluate model
```bash
python main.py evaluate
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_path`: Path to the trained model you have stored. It should be a folder as the one that is created during `train` command. Default: `cache/simple_model/autoencoder`.
- `--data`: Dataset to evaluate the model on. Default: `sadc_2017`.
- `--output`: Output folder to save the outputs. Default: `cache/predictions/`.

The output of this command is two `.csv` files; 
One that contains the metrics as computed per variable/attribute of the data and is stored in the output folder under the name `metrics.csv`,
and one that contains the average metrics of all the variables/attributes of the data and is stored in the output folder under the name `averages.csv`.


### Find outliers
```bash
python main.py find_outliers
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_path`: Path to the trained model you have stored. It should be a folder as the one that is created during `train` command. Default: `cache/simple_model/autoencoder`.
- `--prior`: Prior to use for the variational autoencoder. Choose between two available values: `gaussian` for a Gaussian prior and `gumbel` for applying the gumbel softmax.
- `--data`: Dataset to evaluate the model on. Default: `sadc_2017`.
- `--k`: Weight of the kl loss in the total loss if the model is `VAE`. Default: `1`.
- `--output`: Output folder to save the outputs. Default: `cache/predictions/`.

The output of this command is a `.csv` file that contains the reconstruction loss (or reconstruction, kl, and total loss in case of `VAE`) of each row of the data and is stored in decreasing order in the output folder under the name `errors.csv`.