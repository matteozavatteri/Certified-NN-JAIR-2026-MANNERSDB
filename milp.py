import conf
import gurobipy

import torch
from torch.nn import Linear, ReLU

env = gurobipy.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

def milp_encoding(model, K=1):
	global env
	m = Model("neural network",env=env)

	# dictionaries for (neuron) variables
	m._x = dict() # input
	m._h = dict() # hidden
	m._y = dict() # output

	Layers = list(model._modules.values())

	# input vars
	assert isinstance(Layers[0], torch.nn.Linear)

	for k in range(K):
		for i in range(len(conf.InputFeatures)):
			vtype = GRB.BINARY if i in conf.BinaryIndexes else GRB.CONTINUOUS
			m._x[k,i] = m.addVar(lb=0, ub=1, vtype=vtype, name="x{}_{}".format(k,i))

		w = model.state_dict()['0.weight'].tolist()
		b = model.state_dict()['0.bias'].tolist()
		for i in range(Layers[0].out_features):
			m._h[k,0,i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="h{}_0_{}".format(k,i))
			m.update()
			m.addConstr(m._h[k,0,i] == b[i] + quicksum(w[i][i1] * m._x[k,i1] for i1 in range(Layers[0].in_features)))
			
		# hidden vars
		last = Layers[0]
		j = 1
		for lay in Layers[1:-1]:
			if isinstance(lay, torch.nn.Linear):
				w = model.state_dict()['{}.weight'.format(j)].tolist()
				b = model.state_dict()['{}.bias'.format(j)].tolist()
				for i in range(lay.out_features):
					m._h[k,j,i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="h{}_{}_{}".format(k,j,i))
					m.update()
					m.addConstr(m._h[k,j,i] == b[i] + quicksum(w[i][i1] * m._h[k,j-1,i1] for i1 in range(lay.in_features)))
		
			elif isinstance(lay, torch.nn.ReLU):
				for i in range(last.out_features):
					m._h[k,j,i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="h{}_{}_{}".format(k,j,i))
					m.update()
					m.addConstr(m._h[k,j,i] == max_(m._h[k,j-1,i], constant=0), name="ReLU{}_{}_{}".format(k,j,i))
			else:
				assert False, "unexpected type of layer"

			j += 1
			last = lay

		lay  = Layers[-1]
		j = len(Layers) - 1
		assert isinstance(lay, torch.nn.Linear)
		w = model.state_dict()['{}.weight'.format(j)].tolist()
		b = model.state_dict()['{}.bias'.format(j)].tolist()
		for i in range(len(conf.OutputScores)):
			m._y[k,i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y{}_{}".format(k,i))
			m.update()
			m.addConstr(m._y[k,i] == b[i] + quicksum(w[i][i1] * m._h[k,j-1,i1] for i1 in range(lay.in_features)))

		m.update()
	#m.write("myfile.lp") # for debug

	return m