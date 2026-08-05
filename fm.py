import conf

import properties.P1 as P1
import properties.P2 as P2
import properties.P3 as P3
import properties.P4 as P4
import properties.P14 as P14

import torch.nn as nn

def soft_loss_proxy(network, P, N):	
	L = None
	if P == 1:
		L = P1.soft_loss
	elif P == 2:
		L = P2.soft_loss
	elif P == 3:
		L = P3.soft_loss
	elif P == 4:
		L = P4.soft_loss
	elif P == 14:
		L = P14.soft_loss
	else:
		assert False

	return L(network, N)

def opportunistic_search_proxy(network, args, N=10000):
	f = None
	if args.property == 1:
		f = P1.opportunistic_search
	elif args.property == 2:
		f = P2.opportunistic_search
	elif args.property == 3:
		f = P3.opportunistic_search
	elif args.property == 4:
		f = P4.opportunistic_search
	elif args.property == 14:
		f = P14.opportunistic_search
	else:
		assert False

	return f(network, args, N)

class TwoCopyNet(nn.Module):        # for two-pair properties
    def __init__(self, base_net):
        super().__init__()
        self.net1 = base_net
        self.net2 = base_net

    def forward(self, x1, x2):
        y1 = self.net1(x1)
        y2 = self.net2(x2)
        return y1, y2

def formal_verification_proxy(network, args):
	f = None
	if args.property == 1:
		f = P1.formal_verification
	elif args.property == 2:
		f = P2.formal_verification
	elif args.property == 3:
		f = P3.formal_verification
	elif args.property == 4:
		f = P4.formal_verification
	elif args.property == 14:
		f = P14.formal_verification
	else:
		assert False

	return f(network, args)

def repair_proxy(CEX, CEY, args):
	f = None
	if args.property == 1:
		f = P1.repair
	elif args.property == 4:
		f = P4.repair
	elif args.property == 14:
		f = P14.repair
	else:
		assert False

	return f(CEX, CEY, args)

def sampling_proxy(CEX, CEY, args):
	f = None
	if args.property == 1:
		f = P1.sampling
	elif args.property == 4:
		f = P4.sampling
	elif args.property == 14:
		f = P14.sampling
	else:
		assert False

	return f(CEX, CEY, args)

def repair_proxy_multiple_pairs(D, args):
	f = None
	if args.property == 2:
		f = P2.repair
	elif args.property == 3:
		f = P3.repair
	else:
		assert False

	return f(D, args)

def sampling_proxy_multiple_pairs(D, args, N=None):
	f = None
	if args.property == 2:
		f = P2.sampling
	elif args.property == 3:
		f = P3.sampling
	else:
		assert False

	return f(D, args, N)