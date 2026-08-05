import torch
import torch.nn as nn

import argparse
from statistics import mean
from torch.nn import MSELoss

import conf
import os
import time
import shutil
import json
import sys
import random
import pandas as pd
import conf
import training
import fm

#device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Using device: {}".format(device))
torch.set_default_device(device)

def neural_network(hidden_layers, neurons_per_hidden_layer):
	layers = []
	layers.append(nn.Linear(conf.InputSIZE,neurons_per_hidden_layer))
	layers.append(nn.ReLU())
	
	for _ in range(hidden_layers-1):
		layers.append(nn.Linear(neurons_per_hidden_layer,neurons_per_hidden_layer))
		layers.append(nn.ReLU())

	layers.append(nn.Linear(neurons_per_hidden_layer,conf.OutputSIZE))
	network = nn.Sequential(*layers)
	
	return network

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("property", type=int, default=1, help="(1,2,3,4,14,-2,-3); default: 1; '-P' is property P with repair algorithm that does not compute permutations")
	parser.add_argument("--epsilon", type=float, default=0.0001, help="default: 0.0001")
	parser.add_argument("--learning-rate", type=float, default=0.001, help="default: 0.001")
	parser.add_argument("--epochs", type=int, default=100, help="default: 100")
	parser.add_argument("--early-stopping", type=int, default=100, help="default: 100")
	parser.add_argument("--hidden-layers", type=int, default=7, help="default: 7")
	parser.add_argument("--neurons-per-hidden-layer", type=int, default=8, help="default: 8")
	parser.add_argument("--counterexamples", type=int, default=1, help="max number of counterexamples")
	parser.add_argument("--sampling", type=int, default=0, help="max number of sampled points around repaired counterexamples")
	parser.add_argument("--soft-acceleration", default=False, action='store_true', help="apply soft constraint acceleration")
	parser.add_argument("--keep", default=False, action='store_true', help="always keep the current network (last CEGIS iteration)")
	parser.add_argument("--force", default=False, action='store_true', help="delete previous stat files and do the experiment again")
	parser.add_argument("--seed", type=int, default=1, help="random seed for shuffling dataset (default: 1)")
	parser.add_argument("--cores", type=int, default=4, help="number of physical CPU cores to use where appropriate (e.g., formal verification)")
	args = parser.parse_args()

	assert args.property in [1,2,3,4,14]
	assert args.learning_rate >= 0.0001
	assert args.epsilon >= 0
	assert args.epochs >= 1
	assert args.early_stopping >= 1 and args.early_stopping <= args.epochs
	assert args.hidden_layers >= 1
	assert args.neurons_per_hidden_layer >= 1
	assert args.counterexamples > 0
	assert args.sampling >= 0
	assert args.cores >= 1
	

	if not os.path.exists('experiments'):
		os.mkdir('experiments')
	
	EXPERIMENT = '{property}-{epsilon}-{lr}-{layers}x{neurons}-{counterexamples}-{sampling}-{seed}-{keep}{soft}'.format(
		property=args.property,
		epsilon=args.epsilon,
		lr=args.learning_rate,
		layers=args.hidden_layers,
		neurons=args.neurons_per_hidden_layer,
		counterexamples=args.counterexamples,
		sampling=args.sampling,
		seed=args.seed,
		keep='K' if args.keep else 'L',
		soft='-SA' if args.soft_acceleration else ''
		)
	
	if os.path.exists('experiments/{}'.format(EXPERIMENT)):
		if not args.force:
			exit('experiment exists, use --force to overwrite and do the experiment again')
		else:
			#print('overwriting experiment {}'.format(EXPERIMENT))
			shutil.rmtree('experiments/{}'.format(EXPERIMENT))
	os.mkdir('experiments/{}'.format(EXPERIMENT))
	
	START = time.time()
	df = pd.read_csv("baseline.csv")
	df = df.sample(frac=1, random_state=args.seed)
		
	# partition
	Training = df.iloc[:8840]
	Validation = df.iloc[8840:9945]
	Test = df.iloc[9945:]

	assert len(df) == (len(Training) + len(Validation) + len(Test))

	X_train = torch.tensor(Training.iloc[:,:conf.InputSIZE].values, dtype=torch.float32)
	y_train = torch.tensor(Training.iloc[:,conf.InputSIZE:].values, dtype=torch.float32)
	assert X_train.device.type == device.type
	assert y_train.device.type == device.type

	X_valid = torch.tensor(Validation.iloc[:,:conf.InputSIZE].values, dtype=torch.float32)
	y_valid = torch.tensor(Validation.iloc[:,conf.InputSIZE:].values, dtype=torch.float32)
	assert X_valid.device.type == device.type
	assert y_valid.device.type == device.type

	X_test  = torch.tensor(Test.iloc[:,:conf.InputSIZE].values, dtype=torch.float32)
	y_test  = torch.tensor(Test.iloc[:,conf.InputSIZE:].values, dtype=torch.float32)
	assert X_test.device.type == device.type
	assert y_test.device.type == device.type


	stats = {
		'OriginalInputs' : X_train.shape[0],
		'TrainingTIME' : [],
		'OpportunisticSeachTIME' : [],
		'FormalVerificationTIME' : [],
		'RepairTIME' : [],
		'SamplingTIME' : [],
		'LossOnValidation' : [],
		'LossOnTest' : [],
		'OpportunisticCEX' : [],
		'FormalVerificationCEX' : [],
		'NewPairs' : [],
		'ModifiedExistingPairs' : [],
		'TotalTIME' : None
	}

	loss_fn = MSELoss()
	lossT = None
	cegis_iteration = 0

	network = neural_network(args.hidden_layers, args.neurons_per_hidden_layer)
	
	while True:				
		cegis_iteration += 1	
		# print("iteration={}".format(cegis_iteration))
		if cegis_iteration > 1 and (not args.keep):
			network = neural_network(args.hidden_layers, args.neurons_per_hidden_layer) # new one
		
		soft = args.property if args.soft_acceleration else None
		
		network, lossV, TrainingTIME = training.train(network, X_train, y_train, X_valid, y_valid, args)			


		stats['TrainingTIME'].append(TrainingTIME)
		stats['LossOnValidation'].append(lossV)

		# compute loss on test
		with torch.no_grad():
			y_pred = network(X_test)
			lossT = loss_fn(y_pred, y_test).item()
			stats['LossOnTest'].append(lossT)

		stats['OpportunisticSeachTIME'].append(0)
		stats['OpportunisticCEX'].append(0)
		stats['FormalVerificationTIME'].append(0)
		stats['FormalVerificationCEX'].append(0)
		stats['RepairTIME'].append(0)
		stats['SamplingTIME'].append(0)
		stats['NewPairs'].append(0)
		stats['ModifiedExistingPairs'].append(0)
		stats['CEGISiterations'] = cegis_iteration

		# args.counterexamples = 0 means "baseline" (no certification)
		if args.counterexamples > 0:
			CEX, CEY, OpportunisticTime = fm.opportunistic_search_proxy(network, args, N=1000)
			NumberOfCounterExamples = CEX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX)
			stats['OpportunisticSeachTIME'][-1] = OpportunisticTime
			stats['OpportunisticCEX'][-1] = NumberOfCounterExamples if args.property != 2 and args.property != 3 else int(NumberOfCounterExamples/2)

			if NumberOfCounterExamples == 0:
				CEX, CEY, FormalVerificationTIME = fm.formal_verification_proxy(network, args)
				NumberOfCounterExamples = CEX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX)
				stats['FormalVerificationTIME'][-1] = FormalVerificationTIME
				stats['FormalVerificationCEX'][-1] =  NumberOfCounterExamples if args.property != 2 and args.property != 3 else int(NumberOfCounterExamples/2)
				
			if NumberOfCounterExamples > 0:
				if args.property == 1 or args.property == 4 or args.property == 14:
					REPX, REPY, RepairTIME = fm.repair_proxy(CEX, CEY, args)
					SX, SY, SamplingTIME = fm.sampling_proxy(REPX, REPY, args)
					X_train = torch.cat((X_train, REPX, SX), dim=0) if isinstance(REPX, torch.Tensor) else torch.cat(tuple([X_train] + REPX + SX), dim=0)
					y_train = torch.cat((y_train, REPY, SY), dim=0) if isinstance(REPX, torch.Tensor) else torch.cat(tuple([y_train] + REPY + SY), dim=0)
				elif args.property == 2 or args.property == 3 or args.property == -2 or args.property == -3:
					mask = (X_train[:,conf.index_dtch] <= 0.1) | (X_train[:,conf.index_dtca] <= 0.1)
					X_unstable = torch.cat((X_train[mask],CEX),dim=0) # unstable part of the dataset (this part may be modified)
					y_unstable = torch.cat((y_train[mask],CEY),dim=0) # unstable part of the dataset (this part may be modified)
					D_unstable = torch.cat((X_unstable,y_unstable),dim=1)
					RepairTIME, ModifiedExistingPairs = fm.repair_proxy_multiple_pairs(D_unstable, args)
					stats['ModifiedExistingPairs'][-1] = ModifiedExistingPairs - CEX.shape[0]
					SX, SY, SamplingTIME, AddedPairs = fm.sampling_proxy_multiple_pairs(D_unstable, args, N=int(CEX.shape[0]/2))
					X_train = torch.cat((X_train[~mask], X_unstable, SX), dim=0)
					y_train = torch.cat((y_train[~mask], y_unstable, SY), dim=0)
				else:
					assert False
				stats['RepairTIME'][-1] = RepairTIME
				stats['NewPairs'][-1] = CEX.shape[0] + SX.shape[0] if isinstance(CEX, torch.Tensor) else sum(cex.shape[0] for cex in CEX) + sum(sx.shape[0] for sx in SX)
				stats['SamplingTIME'][-1] = SamplingTIME

		# statistics
		with open('experiments/{}/stats.json'.format(EXPERIMENT), 'w') as f:
			json.dump(stats, f)

		torch.save(network.state_dict(), 'experiments/{}/network-{}.pt'.format(EXPERIMENT,cegis_iteration))
		
		if NumberOfCounterExamples == 0:
			#print("No counterexamples found")
			break

	stats['TotalTIME'] = time.time() - START

	
	# final statistics
	with open('experiments/{}/stats.json'.format(EXPERIMENT), 'w') as f:
		json.dump(stats, f)