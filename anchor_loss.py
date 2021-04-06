import numpy as np
import torch


class AnchorLoss(torch.nn.Module):
	def __init__(self, var_dict, w0_dict):
		super(AnchorLoss, self).__init__()
		self.var_dict = var_dict
		self.w0_dict = w0_dict
		self.keys = self.var_dict.keys()

	def forward(self, w_dict):
		loss = 0		
		for key in self.keys: #for each w, b of each layer
			var = self.var_dict[key]
			w0 = self.w0_dict[key]
			w = w_dict[key]

			loss = loss + torch.sum((w - w0)**2)/var	

		loss = torch.sum(loss)
		return loss

'''
v_dict = {}
w0_dict = {}
w_dict = {}

keys = ['layer1', 'layer2', 'layer3']
for key in keys:
	
	v_dict[key] = torch.ones(1)
	w0_dict[key] = 2*torch.ones(3)
	w_dict[key] = 3*torch.ones(3)

print(v_dict)

AnchLoss = AnchorLoss(v_dict, w0_dict)

print(AnchLoss.forward(w_dict))
'''
