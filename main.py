import torch
import torch.optim as optim
import numpy as np 
from BNN import Net
from gaussian_nll_loss_class import GaussianNLLLoss
from anchor_loss import AnchorLoss
from RMSE import RMSELoss
import datetime
import time
import argparse
import os


# Parse command line args
p = argparse.ArgumentParser()
p.add_argument('-s', '--seed', type=int, required=False)
p.add_argument('-c', '--checkpt', type=int, required=False)
args = p.parse_args()

# Set device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
print(f'Device is: {device}')

# Set seeds for reproducibility
if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DST_DIR = 'output_{}'.format(args.seed)
else:
    np.random.seed(0)
    torch.manual_seed(0)
    DST_DIR = 'output'

if not os.path.isdir(DST_DIR):
    os.mkdir(DST_DIR)

# Load data
print('Loading data start: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
data = np.load('X_y.npz')
print('Loading X and y end: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
X = data['X']
print('Loading X into X end: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
y = data['y']
print('Loading y into y end: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
N = X.shape[0]
del data
print('Loading data end: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('\nNumber of data vectors: ' + str(N))

# Create index arrays
inds = np.arange(N)
rand_inds = np.random.permutation(inds)

# Create train and test sets
train_size = int(0.8*N)
test_size = N - train_size
train_inds = rand_inds[:train_size]
test_inds = rand_inds[train_size:]
X_train = torch.from_numpy(X[train_inds]).to(device)
X_test = torch.from_numpy(X[test_inds]).to(device)
del X
y_train = torch.from_numpy(y[train_inds]).to(device)
y_test = torch.from_numpy(y[test_inds]).to(device)
del y

# Hyperparams
batch_size = 2048
lr = 1e-3
n_epochs = 10000
n_train_eval_per_epoch = int(np.ceil(train_size/batch_size))
#n_test_eval_per_epoch = int(np.ceil(test_size/batch_size))
n_test_eval_per_epoch = 5

# Load net
net = Net().double().to(device)
start_epoch = 0
if args.checkpt:
    start_epoch = int(args.checkpt)
    net.load_state_dict(torch.load(f'{DST_DIR}/mymodel_{args.seed}_{start_epoch}'))
    print('Loaded model: {}'.format(args.checkpt))


# Optimiser
optimiser = optim.Adam(net.parameters(), lr=lr)

# Anchor loss
anchor_params = {}
anchor_vars = {}
for name, p in net.get_anchor_params().items():
    anchor_params[name] = p.detach().clone()

for name, p in net.get_anchor_vars().items():
    anchor_vars[name] = p.detach().clone()

anch_loss = AnchorLoss(anchor_vars, anchor_params).to(device)

# Gaussian NLL Loss
gnll_loss = GaussianNLLLoss(reduction='sum').to(device)

# RMSE (for comparisons)
rmse_loss = RMSELoss().to(device)

# Train model
print('\nTraining start: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for epoch in range(start_epoch, n_epochs):
    
    # Train Loop
    train_loss_tracker = 0
    a_loss_tracker = 0
    for i in range(n_train_eval_per_epoch):
        optimiser.zero_grad()
        X_i = X_train[i*batch_size:(i+1)*batch_size, :]
        y_i = y_train[i*batch_size:(i+1)*batch_size, :]
        out_i, var_i = net(X_i)
        params = net.get_params()
        
        g_loss = 2*gnll_loss(out_i, y_i, var_i)/X_i.shape[0]
        a_loss = anch_loss(params)/train_size
        train_loss = g_loss + a_loss
        train_loss.backward()
        optimiser.step()
        a_loss_tracker += a_loss.item()
        train_loss_tracker += train_loss.item()
        
    train_loss_tracker /= n_train_eval_per_epoch
    a_loss_tracker /= n_train_eval_per_epoch

    # Test Loop
    test_loss_tracker = 0
    targs = []
    preds = []
    for i in range(n_test_eval_per_epoch):
        X_i = X_test[i*batch_size:(i+1)*batch_size, :]
        y_i = y_test[i*batch_size:(i+1)*batch_size, :]
        out_i, var_i = net(X_i)
        test_loss = 2*gnll_loss(out_i, y_i, var_i)/X_i.shape[0]
        test_loss_tracker += test_loss.item()
    
    test_loss_tracker = test_loss_tracker/n_test_eval_per_epoch + a_loss_tracker

    # Save model and report losses
    if (epoch+1)%20 == 0:
        rmse = rmse_loss(out_i, y_i).item()
        torch.save(net.state_dict(), '{}/mymodel_{}_{}'.format(DST_DIR, args.seed, epoch+1))
        print(f'[epoch {epoch+1}] train: {round(train_loss_tracker,4)},\ttest: {round(test_loss_tracker,4)},\tanch: {round(a_loss_tracker,4)},\tRMSE: {round(rmse,4)},\t'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        print(f'[epoch {epoch+1}] train: {round(train_loss_tracker,4)},\ttest: {round(test_loss_tracker,4)},\tanch: {round(a_loss_tracker,4)},\t'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


    # Shuffle indices
    if (epoch+1)%100 == 0:
        print(f'Epoch {epoch+1}: shuffle at ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train_inds = np.random.permutation(np.arange(train_size))
        test_inds = np.random.permutation(np.arange(test_size))
        X_train = X_train.to('cpu')
        X_train = X_train[train_inds]
        X_train = X_train.to(device)
        X_test = X_test.to('cpu')
        X_test = X_test[test_inds]
        X_test = X_test.to(device)
        y_train = y_train.to('cpu')
        y_train = y_train[train_inds]
        y_train = y_train.to(device)
        y_test = y_test.to('cpu')
        y_test = y_test[test_inds]
        y_test = y_test.to(device)
