# ====================================================================
# Property 3
# ====================================================================
# If the distance to the closest animal is within 0.6 (meters), then 
# the adequateness of mopping the floor must be less than or equal to 
# the adequateness of mopping the floor in all possible scenarios in 
# which the distance to the closest human is within 0.6 (meters).
# ====================================================================

import time
import random
from maraboupy import Marabou, MarabouCore
from maraboupy import MarabouUtils

import math
import gurobipy as gp
import os
import torch
import fm
import tempfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import conf

def precondition_enabling_batch(N):
	#print("precondition_enabling_batch P3")
	low  = torch.zeros(conf.InputSIZE)
	high = torch.ones(conf.InputSIZE)
	
	high[conf.index_dtca] = 0.1
	xA = low + (high - low) * torch.rand(N, low.shape[0])
	xA[:,conf.BinaryIndexes] = torch.randint(low=0, high=2, size=(N, len(conf.BinaryIndexes)), dtype=xA.dtype)

	high[conf.index_dtca] = 1
	high[conf.index_dtch] = 0.1
	xH = low + (high - low) * torch.rand(N, low.shape[0])
	xH[:,conf.BinaryIndexes] = torch.randint(low=0, high=2, size=(N, len(conf.BinaryIndexes)), dtype=xH.dtype)

	return xA, xH

def soft_loss(network, N):
	#print("soft_loss P3")
	xA, xH = precondition_enabling_batch(N)
	yA = network(xA)[:, conf.index_mop]
	yH = network(xH)[:, conf.index_mop]
	VS = torch.relu(yA - yH)
	mask = VS > 0
	return VS[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)


def opportunistic_search(network, args, N):
	#print("opportunistic_search P3")
	start = time.time()
	xA, xH = precondition_enabling_batch(N)
	with torch.inference_mode():
		yA = network(xA)
		yH = network(xH)

	mask = yA[:,conf.index_mop] >= yH[:,conf.index_mop] + args.epsilon
	xA = xA[mask][:args.counterexamples]
	xH = xH[mask][:args.counterexamples]
	yA = yA[mask][:args.counterexamples]
	yH = yH[mask][:args.counterexamples]
	
	CEX = torch.cat((xA,xH),dim=0)
	CEY = torch.cat((yA,yH),dim=0)
	OpportunisticSeachTIME = time.time() - start
	return CEX, CEY, OpportunisticSeachTIME

def repair(D, args):
	#print("repair P3")
	start = time.time()	
	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()
	m = gp.Model("repair",env=env)

	S = dict()
	for row in D:
		key = tuple(row.tolist())
		if key not in S:
			S[key] = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
	
	maskA = D[:,conf.index_dtca] <= 0.1
	maskH = D[:,conf.index_dtch] <= 0.1
	RowsA = D[maskA]	
	RowsH = D[maskH]

	mop = conf.InputSIZE + conf.index_mop
	for rA in RowsA:
		keyA = tuple(rA.tolist())
		for rH in RowsH:
			keyH = tuple(rH.tolist())
			m.addConstr(rA[mop].item() + S[keyA] <= rH[mop].item() + S[keyH])

	m.setObjective(gp.quicksum(s ** 2 for s in S.values()), gp.GRB.MINIMIZE)

	m.optimize()
	assert m.Status == gp.GRB.OPTIMAL

	
	for i in range(D.shape[0]):
		key = tuple(D[i].tolist())
		D[i][mop] += S[key].X

	ModifiedPairs = sum(1 for s in S.values() if s.X != 0)
	m.dispose()
	
	return time.time() - start, ModifiedPairs


def sampling(D, args, N):
	#print("sampling P3")
	start = time.time()
	REX = D[-2*N:,:conf.InputSIZE]
	REY = D[-2*N:,conf.InputSIZE:]
	x = REX.repeat_interleave(args.sampling,0)
	y = REY.repeat_interleave(args.sampling,0)

	noise = 0.1 * torch.randn(args.sampling * 2*N,len(conf.ContinuousIndexes))
	x[:,conf.ContinuousIndexes] += noise
	
	noise = 0.1 * torch.randn(args.sampling * 2*N,conf.OutputSIZE)
	y += noise


	# those pairs that can combine with another to activate the precondition
	critical_mask = (x[:,conf.index_dtca] <= 0.1) | (x[:,conf.index_dtch] <= 0.1)  
	SX = x[critical_mask]
	SY = y[critical_mask]
	
	# first step: find pairs violating the property among sampled ones
	remove_mask = torch.ones(SX.shape[0], dtype=torch.bool)

	for a in range(SX.shape[0]):
		for h in range(SX.shape[0]):
			if SX[a][conf.index_dtca].item() <= 0.1 and SX[h][conf.index_dtch].item() <= 0.1 and SY[a][conf.index_mop].item() > SY[h][conf.index_mop].item():
				remove_mask[a] = False
				remove_mask[h] = False

	SX = SX[remove_mask]
	SY = SY[remove_mask]

	# second step: find pairs in conjunction with dataset that violate property (sampled, dataset)
	remove_mask = torch.ones(SX.shape[0], dtype=torch.bool)
	for a in range(SX.shape[0]):
		for h in range(D.shape[0]):
			if SX[a][conf.index_dtca].item() <= 0.1 and D[h][conf.index_dtch].item() <= 0.1 and SY[a][conf.index_mop].item() > D[h][conf.index_mop].item():
				remove_mask[a] = False

	SX = SX[remove_mask]
	SY = SY[remove_mask]

	# third step: find pairs in conjunction with dataset that violate property (dataset, sampled)
	remove_mask = torch.ones(SX.shape[0], dtype=torch.bool)
	for a in range(D.shape[0]):
		for h in range(SX.shape[0]):
			if D[a][conf.index_dtch].item() <= 0.1 and SX[h][conf.index_dtca].item() <= 0.1 and D[a][conf.index_mop].item() > SY[h][conf.index_mop].item():
				remove_mask[h] = False

	SX = SX[remove_mask]
	SY = SY[remove_mask]

	# those that can't combine with any other (both human and animal not close)
	SX = torch.cat((SX,x[~critical_mask]), dim=0)
	SY = torch.cat((SY,y[~critical_mask]), dim=0)

	return SX, SY, time.time() - start, SX.shape[0] - D.shape[0]


def formal_verification(network, args):
	#print("formal_verification P3")
	start = time.time()
	Q = None
	with tempfile.TemporaryDirectory() as tmpdir:
		network.eval()
		joint_model = fm.TwoCopyNet(network)
		xA = torch.zeros(1, conf.InputSIZE)
		xH = torch.zeros(1, conf.InputSIZE)
		joint_model.eval()
		onnx_path = os.path.join(tmpdir, "model.onnx")
		with torch.inference_mode():		
			torch.onnx.export(joint_model,(xA, xH), onnx_path)
		Q = Marabou.read_onnx(onnx_path)

	assert Q is not None

	XA = Q.inputVars[0][0]
	XH = Q.inputVars[1][0]
	YA = Q.outputVars[0][0]
	YH = Q.outputVars[1][0]

	for i in range(conf.InputSIZE):
		Q.setLowerBound(XA[i], 0)
		Q.setUpperBound(XA[i], 1)
		Q.setLowerBound(XH[i], 0)
		Q.setUpperBound(XH[i], 1)

	Q.setUpperBound(XA[conf.index_dtch], 0.1)
	Q.setUpperBound(XH[conf.index_dtca], 0.1)

	for X in [XA, XH]:
		for idx in conf.BinaryIndexes:
			disjunction = []
			for v in [0, 1]:
				eq = MarabouUtils.Equation(MarabouCore.Equation.EQ)
				eq.addAddend(1, X[idx])
				eq.setScalar(v)
				disjunction.append([eq])
			Q.addDisjunctionConstraint(disjunction)

	eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
	eq.addAddend(1, YA[conf.index_mop])
	eq.addAddend(-1, YH[conf.index_mop])
	eq.setScalar(args.epsilon)
	Q.addEquation(eq)

	options = Marabou.createOptions(numWorkers=args.cores,verbosity=0)
	sat, vals, stats = Q.solve(verbose=False, options=options)

	if sat == 'sat':
		CEXA = torch.tensor([[vals[v] for v in XA]])
		CEYA = torch.tensor([[vals[v] for v in YA]])
		CEXH = torch.tensor([[vals[v] for v in XH]])
		CEYH = torch.tensor([[vals[v] for v in YH]])
	else:
		CEXA = torch.empty(0, conf.InputSIZE)
		CEYA = torch.empty(0, conf.OutputSIZE)
		CEXH = torch.empty(0, conf.InputSIZE)
		CEYH = torch.empty(0, conf.OutputSIZE)
		
	CEX = torch.cat((CEXA,CEXH),dim=0)
	CEY = torch.cat((CEYA,CEYH),dim=0)

	FormalVerificationTIME = time.time() - start
	return CEX, CEY, FormalVerificationTIME