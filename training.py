import torch
import torch.optim as optim
import time
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import copy
import conf
import fm

def train(network, X_train, y_train, X_valid, y_valid, args):
	start = time.time()
	optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
	batch_size = 100

	loss_fn = nn.MSELoss()
	best_state = copy.deepcopy(network.state_dict())
	best_lossV  = 1000000000
	best_lossS  = 1000000000 if args.soft_acceleration else 0
	
	#print("Initial network: loss(validation) = {}".format(best_lossV))

	stop = 0

	train = torch.utils.data.TensorDataset(X_train, y_train)
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

	for epoch in range(args.epochs):
		lossS = 0
		#print("epoch = {}".format(epoch))
		for data in train_loader:
			Xbatch, ybatch = data
			y_pred = network(Xbatch)
			loss = loss_fn(y_pred, ybatch)
			# soft constraint loss
			if args.soft_acceleration:
				lossS = fm.soft_loss_proxy(network, args.property, batch_size) # random soft batch
				loss += lossS
								
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		# compute loss on valid
		lossS = 0
		with torch.no_grad():
			y_pred = network(X_valid)
			lossV = loss_fn(y_pred, y_valid).item()
			if args.soft_acceleration:
				lossS = fm.soft_loss_proxy(network, args.property, X_valid.shape[0]).item()
				
		if (lossV + lossS) < (best_lossV + best_lossS): # see above
			best_state = copy.deepcopy(network.state_dict())
			best_lossV  = lossV
			best_lossS  = lossS
			#print("Update after epoch {}: lossV = {}, lossS = {}".format(epoch, best_lossV, best_lossS))
			stop = 0
		else:
			stop += 1

		if stop > args.early_stopping:
			break

	network.load_state_dict(best_state)
	return network, best_lossV, time.time() - start