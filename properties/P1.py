# ====================================================================
# Property 1
# ====================================================================
# If the distance to the closest child is <= 0.6 meter (0.1 normalized),
# then mopping the floor has minimal score.
# ====================================================================

import time
import random
from maraboupy import Marabou, MarabouCore
from maraboupy import MarabouUtils

import tempfile
import gurobipy as gp

import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import conf

def precondition_enabling_batch(N):
	#print("precondition_enabling_batch P1")
	low  = torch.zeros(conf.InputSIZE)
	high = torch.ones(conf.InputSIZE)
	high[conf.index_dtcc] = 0.1
	x = low + (high - low) * torch.rand(N, low.shape[0])
	x[:,conf.BinaryIndexes] = torch.randint(low=0, high=2, size=(N, len(conf.BinaryIndexes)), dtype=x.dtype)
	return x

def soft_loss(network, N):	
	#print("soft_loss P1")
	x = precondition_enabling_batch(N)
	y = network(x)
	VS = torch.relu(y[:, conf.index_mop].unsqueeze(1) - y[:, conf.ActionsDifferentFromMop]).sum(dim=1)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P1")
	start = time.time()
	x = precondition_enabling_batch(N)
	with torch.inference_mode():
		y = network(x)

	mask = y[:, conf.index_mop] >= y[:, conf.ActionsDifferentFromMop].min(dim=1).values + args.epsilon

	CEX = x[mask][:args.counterexamples]
	CEY = y[mask][:args.counterexamples]
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def repair(CEX, CEY, args):
	#print("repair P1")
	start = time.time()
	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()

	for i in range(CEY.shape[0]):
		m = gp.Model("repair",env=env)
		S = [m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for _ in range(conf.OutputSIZE)]
		m.update()
		for act in conf.ActionsDifferentFromMop:
			m.addConstr(CEY[i][conf.index_mop].item() + S[conf.index_mop] <= CEY[i][act].item() + S[act])
			
		m.setObjective(gp.quicksum(s ** 2 for s in S), gp.GRB.MINIMIZE)
		m.update()
		m.optimize()
		assert m.Status == gp.GRB.OPTIMAL

		for act in range(conf.OutputSIZE):
			CEY[i][act] += S[act].X

		m.dispose()
	RepairTIME = time.time() - start
	return CEX, CEY, RepairTIME


def sampling(REPX, REPY, args):
	#print("sampling P1")
	start = time.time()
	N = REPX.shape[0] # actual counterexamples found
	x = REPX.repeat_interleave(args.sampling,0)
	y = REPY.repeat_interleave(args.sampling,0)

	noise = 0.1 * torch.randn(args.sampling * N,len(conf.ContinuousIndexes))
	x[:,conf.ContinuousIndexes] += noise
	
	noise = 0.1 * torch.randn(args.sampling * N,conf.OutputSIZE)
	y += noise

	not_pre = x[:,conf.index_dtcc] > 0.1
	post = y[:, conf.index_mop] <= y[:, conf.ActionsDifferentFromMop].min(dim=1).values
	mask = not_pre | post
	
	return x[mask], y[mask], time.time() - start

def formal_verification(network, args):
	#print("formal_verification P1")
	start = time.time()
	
	Q = None
	with tempfile.TemporaryDirectory() as tmpdir:
		network.eval()
		x = torch.zeros(1, conf.InputSIZE)
		onnx_path = os.path.join(tmpdir, "model.onnx")
		with torch.inference_mode():		
			torch.onnx.export(network, x, onnx_path)
		Q = Marabou.read_onnx(onnx_path)

	assert Q is not None

	X, Y = Q.inputVars[0][0], Q.outputVars[0][0]

	for i in range(conf.InputSIZE):
		Q.setLowerBound(X[i], 0)
		Q.setUpperBound(X[i], 1)
	
	Q.setUpperBound(X[conf.index_dtcc], 0.1)			
	
	for x in conf.BinaryIndexes:
		disjunction = []
		for v in [0,1]:
			eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
			eq.addAddend(1, X[x])
			eq.setScalar(v)
			disjunction.append([eq])
		Q.addDisjunctionConstraint(disjunction)

	disjunction = []	
	for act in conf.ActionsDifferentFromMop:
		eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
		eq.addAddend(1, Y[conf.index_mop])
		eq.addAddend(-1, Y[act])
		eq.setScalar(args.epsilon)
		disjunction.append([eq])
	Q.addDisjunctionConstraint(disjunction)

	options = Marabou.createOptions(numWorkers=args.cores, verbosity=0)
	sat, vals, stats = Q.solve(verbose=False,options=options)
	CEX = torch.tensor([[vals[x] for x in X]]) if sat == 'sat' else torch.empty(0,conf.InputSIZE)
	CEY = torch.tensor([[vals[y] for y in Y]]) if sat == 'sat' else torch.empty(0,conf.OutputSIZE)
	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME