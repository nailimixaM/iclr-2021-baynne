# iclr-2021-baynne
ICLR 2021 Deep Learning for Simulation workshop paper.

# Files
- main.py: main file, sets hyperparams, performs training, saves model checkpoints etc
- BNN.py: class definition of a heteroscedastic neural network
- gaussian_nll_loss_class.py: Gaussian negative log-likelihood loss 
- anchor_loss.py: anchor loss
- RMSE.py: root mean square error loss

# Basic usage
Train a single neural network
- python main.py (default: seed = 0, start epoch = 0)
- python main.py -s 42 (seed the training process with seed = 42) 
- python main.py -s 42 -c 1000 (start the training process from model pre-saved at epoch = 1000)

# Ensemble training
- An ensemble can be trained by uniquely seeding each member of the ensemble and running 'python main.py -s (seed)'

# Requirements
- python 3.6.8
- pytorch 1.6.0
