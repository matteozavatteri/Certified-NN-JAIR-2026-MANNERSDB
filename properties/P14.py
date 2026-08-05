# ====================================================================
# Property 1 AND Property 4
# ====================================================================

import time
import random
from maraboupy import Marabou, MarabouCore
from maraboupy import MarabouUtils

import gurobipy as gp

import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

import properties.P1 as P1
import properties.P4 as P4

import conf

def soft_loss(network, x):	
	#print("soft_loss P14")
	return P1.soft_loss(network, x) + P4.soft_loss(network, x)

def opportunistic_search(network, args, N):
	#print("opportunistic_search P14")
	CEX1, CEY1, OpportunisticTime1 = P1.opportunistic_search(network, args, N)
	CEX4, CEY4, OpportunisticTime4 = P4.opportunistic_search(network, args, N)
	return [CEX1,CEX4], [CEY1,CEY4], OpportunisticTime1 + OpportunisticTime4

def repair(CEX, CEY, args): 
	#print("repair P14")
	REPX1, REPY1, RepairTIME1 = P1.repair(CEX[0], CEY[0], args)
	REPX4, REPY4, RepairTIME4 = P4.repair(CEX[-1], CEY[-1], args)
	return [REPX1, REPX4], [REPY1, REPY4], RepairTIME1 + RepairTIME4

def sampling(REPX, REPY, args):
	#print("sampling P14")
	SX1, SY1, SamplingTIME1 = P1.sampling(REPX[0], REPY[0], args) 
	SX4, SY4, SamplingTIME4 = P4.sampling(REPX[-1], REPY[-1], args) 
	return [SX1,SX4], [SY1,SY4], SamplingTIME1 + SamplingTIME4 

def formal_verification(network, args):
	#print("formal_verification P14")
	CEX1, CEY1, FormalVerificationTIME1 = P1.formal_verification(network, args)
	CEX4, CEY4, FormalVerificationTIME4 = P4.formal_verification(network, args)
	return [CEX1,CEX4], [CEY1,CEY4], FormalVerificationTIME1 + FormalVerificationTIME4