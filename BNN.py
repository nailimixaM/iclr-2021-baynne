# Set up the NN
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):

        def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(900,900)
                self.fc2 = nn.Linear(900,900)
                self.fc3 = nn.Linear(900,900)
                self.fc4 = nn.Linear(900,900)
                self.fc5 = nn.Linear(900,900)
                self.fc6 = nn.Linear(900,6) #targ
                self.fc6b = nn.Linear(900,6) #var est for each targ

                #Kaiming initialisation
                nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
                nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
                nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
                nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
                nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='relu')
                nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='relu')

                #Anchor variances
                self.anchor_vars = self.get_param_vars()

                #Anchor weights & biases
                self.anchor_params = self.get_params()


        def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                xa = torch.sigmoid(self.fc6(x))
                xb = torch.exp(self.fc6b(x))
                return xa, xb

        def get_param_tensor(self, param_name):
		#param_name can be fcX.weight or fcX.bias
                for name, param in self.named_parameters():
                        if name == param_name:
                                print(name, param.size())
                                return param

        def get_params(self):
                params = {}
                for name, ps in self.named_parameters():
                        params[name] = ps

                return params

        def get_param_vars(self):
                param_vars = {}
                for name, params in self.named_parameters():
                        param_vars[name] = torch.std(params)**2

                return param_vars
                                	
        def get_anchor_params(self):
                return self.anchor_params

        def get_anchor_vars(self):
                return self.anchor_vars
