# iclr-2021-baynne
ICLR 2021 Deep Learning for Simulation workshop paper.

# Summary
The paper applies a Bayesian neural network ensemble method for regression of parameters of a flame edge model. The ensemble is trained on synthetic data and evaluated on real experiments of a Bunsen flame.    

# Files
- [main.py](https://github.com/nailimixaM/iclr-2021-baynne/blob/main/main.py): main file, sets hyperparams, performs training, saves model checkpoints etc
- [BNN.py](https://github.com/nailimixaM/iclr-2021-baynne/blob/main/BNN.py): class definition of a heteroscedastic neural network
- [gaussian_nll_loss_class.py](https://github.com/nailimixaM/iclr-2021-baynne/blob/main/gaussian_nll_loss_class.py): Gaussian negative log-likelihood loss 
- [anchor_loss.py](https://github.com/nailimixaM/iclr-2021-baynne/blob/main/anchor_loss.py): anchor loss
- [RMSE.py](https://github.com/nailimixaM/iclr-2021-baynne/blob/main/RMSE.py): root mean square error loss

# Training data sets
- [X_y_full.npz](https://drive.google.com/file/d/1001fCJ6gNCw6O7DW4MyQrsroKiGcLjyo/view?usp=sharing) (~11 GB)
- [X_y.npz](https://drive.google.com/file/d/1_ECkIObb3mNL7BMmYjO0MwAyBDnof_ka/view?usp=sharing) (~2 GB, for debugging purposes)

# Real experiments data set
- [X_experiments.npz](https://drive.google.com/file/d/19HT2Fq4Az-6GlW3w87lM-ty6GdjGOvOR/view?usp=sharing) (~14 MB)

# Basic usage
To train a single neural network
- python main.py (default: seed = 0, start epoch = 0, models saved to directory 'output_0')
- python main.py -s 1 (seed the training process with seed = 1, models saved to 'output_1') 
- python main.py -s 1 -c 1000 (start the training process from model pre-saved at epoch = 1000)

A neural network with seed = 1 pre-trained for 1000 epochs is provided in 'output_1'.

# Ensemble training
- An ensemble can be trained by uniquely seeding each member of the ensemble and running 'python main.py -s (seed)'

# Requirements
- python 3.6.8
- pytorch 1.6.0
