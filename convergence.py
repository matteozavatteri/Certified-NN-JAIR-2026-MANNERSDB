import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import argparse
import conf

import experiments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def neural_network(hidden_layers, neurons_per_hidden_layer):
	InputNeurons  = conf.InputSIZE
	OutputNeurons = conf.OutputSIZE
	layers = []
	layers.append(nn.Linear(InputNeurons,neurons_per_hidden_layer))
	layers.append(nn.ReLU())
	for _ in range(hidden_layers-1):
		layers.append(nn.Linear(neurons_per_hidden_layer,neurons_per_hidden_layer))
		layers.append(nn.ReLU())
	layers.append(nn.Linear(neurons_per_hidden_layer,OutputNeurons))
	network = nn.Sequential(*layers)
	return network

import os 
import json
from statistics import mean, stdev

index_dtcc = conf.index_dtcc
index_dtch = conf.index_dtch
index_dtca = conf.index_dtca

index_mop         = conf.index_mop         
index_warm_food   = conf.index_warm_food   
index_big_objects = conf.index_big_obj

def CounterExamples(P,network,N=1000):
	
	low = torch.zeros(conf.InputSIZE)
	high = torch.ones(conf.InputSIZE)

	CE = 0

	if P == 1:	
		high[index_dtcc] = 0.1
		x = low + (high - low) * torch.rand(N, low.shape[0])
		y = network(x)
		row_min = y.min(dim=1).values
		mask = y[:, index_mop] >= row_min + experiments.EPSILON[0]
		CE = mask.sum().item()

	elif P == 2:
		high[index_dtch] = 0.1
		xH = low + (high - low) * torch.rand(N, low.shape[0])
		yH = network(xH)

		high[index_dtch] = 1
		high[index_dtca] = 0.1
		xA = low + (high - low) * torch.rand(N, low.shape[0])
		yA = network(xA)

		mask = yH[:, index_mop] >= yA[:, index_mop] + experiments.EPSILON[0]
		CE = mask.sum().item()

	elif P == 3:
		high[index_dtca] = 0.1
		xA = low + (high - low) * torch.rand(N, low.shape[0])
		yA = network(xA)

		high[index_dtca] = 1
		high[index_dtch] = 0.1
		xH = low + (high - low) * torch.rand(N, low.shape[0])
		yH = network(xH)

		mask = yA[:, index_mop] >= yH[:, index_mop] + experiments.EPSILON[0]
		CE = mask.sum().item()

	elif P == 4:
		low[index_dtcc] = 0.75
		x = low + (high - low) * torch.rand(N, low.shape[0])
		y = network(x)

		row_max = y.max(dim=1).values
		mask = (row_max >= y[:, index_warm_food] + experiments.EPSILON[0]) & (row_max >= y[:, index_big_objects] + experiments.EPSILON[0])
		CE = mask.sum().item()

	elif P == 14:
		CE = CounterExamples(1,network) + CounterExamples(4,network)

	return CE

if not os.path.exists("img"):
	os.mkdir("img")



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--baseline", default=False, action='store_true')
	parser.add_argument("--counterexamples", default=False, action='store_true')
	parser.add_argument("--sampling", default=False, action='store_true')

	
	args = parser.parse_args()

	assert int(args.baseline) + int(args.counterexamples) + int(args.sampling) == 1

	COUNTEREXAMPLES = [25,50,75] if args.counterexamples else [50]
	SAMPLING = [10,20,30] if args.sampling else [20]

	if args.baseline:
		COLORS  = {
			'7x9'   : 'tab:orange',
			'8x8'   : 'tab:red',
			'9x7'   : 'tab:green',
			'10x6'  : 'tab:blue',
			'11x10' : 'tab:purple',
			}

		MARKERS  = {
			'7x9'   : 'o',
			'8x8'   : 's',
			'9x7'   : 'X',
			'10x6'  : 'P',
			'11x10' : 'D',
			}

	elif args.sampling:
		COLORS  = {
				10   : 'tab:red',
				20   : 'tab:blue',
				30   : 'tab:green',
				-1  : 'tab:orange',
				-2 : 'tab:purple',
				}

		MARKERS  = {
				10   : 'o',
				20   : 's',
				30   : 'X',
				-1  : 'P',
				-2 : 'D',
				}

	elif args.counterexamples:
		COLORS  = {
			25   : 'tab:red',
			50   : 'tab:blue',
			75   : 'tab:green',
			-1  : 'tab:orange',
			-2 : 'tab:purple',
			}

		MARKERS  = {
			25   : 'o',
			50   : 's',
			75   : 'X',
			-1  : 'P',
			-2 : 'D',
			}


	raw_data = experiments.load_experiment_data()

	xticks = 11

	print("Computing convergence ...")
	DATA = {k : [] for k in raw_data if raw_data[k]['TotalTIME'] is not None}
	for key in DATA:
		P = int(key.split('-')[0])
		path = 'experiments/{}'.format(key)
		INDEXES = [round(i * (raw_data[key]['CEGISiterations'] - 1) / (xticks - 1)) for i in range(xticks)]
		for index in INDEXES:
			network = neural_network(10,6)
			network.load_state_dict(torch.load('{}/network-{}.pt'.format(path,index+1),map_location=torch.device(device)))
			with torch.inference_mode():
				DATA[key].append(CounterExamples(P,network))

	print("done.")

	x = [10*i for i in range(xticks)]

	print("Computing plots...")

	for P in experiments.PROPERTIES:
		print('Property P{}'.format(P))
		fig, ax = plt.subplots()
		plt.xlabel('CEGIS iterations')
		labels = {'xticks' : x, 'xticklabels': x}
		ax.set(**labels)
		for SOFT in ['','-SA']:
			for CEX in COUNTEREXAMPLES:
				for SAM in SAMPLING:
					KEYS = [experiments.key_template.format(
						property=P, 
					    epsilon=epsilon, 
					    learning_rate=learning_rate, 
					    layers=10,
					    neurons=6,
					    counterexamples=CEX, 
					    sampling=SAM, 
					    seed=seed, 
					    soft=SOFT
					    ) \
					    for epsilon in experiments.EPSILON \
					    for learning_rate in experiments.LEARNING_RATES \
					    for seed in experiments.SEEDS]

					y = []
					for i in range(xticks):
						y.append(mean(DATA[k][i] for k in KEYS if k in DATA))

					linestyle = 'solid' if SOFT == '' else 'dashed'
					if args.baseline:
						N = '10x6'
						label = N if SOFT == '' else None
						ax.plot(x,y,marker=MARKERS[N],color=COLORS[N],linewidth=2,linestyle=linestyle,label=label)
						plt.ylabel('Baseline')
					elif args.counterexamples:
						label = CEX if SOFT == '' else None
						ax.plot(x,y,marker=MARKERS[CEX],color=COLORS[CEX],linewidth=2,linestyle=linestyle,label=label)
						plt.ylabel('Counterexamples')
					elif args.sampling:
						label = SAM if SOFT == '' else None
						ax.plot(x,y,marker=MARKERS[SAM],color=COLORS[SAM],linewidth=2,linestyle=linestyle,label=label)
						plt.ylabel('Sampling')
			
		ax.xaxis.set_major_formatter(PercentFormatter(100))
		plt.legend()
		plt.grid(True)

		ablation = None
		if args.baseline:
			ablation = 'baseline'
		elif args.counterexamples:
			ablation = 'counterexamples'
		else:
			assert args.sampling
			ablation = 'sampling'

		print('Saving img/convergence-robot-{}-P{}.pdf'.format(ablation,P))
		plt.savefig('img/convergence-robot-{}-P{}.pdf'.format(ablation,P))

	print("done")
