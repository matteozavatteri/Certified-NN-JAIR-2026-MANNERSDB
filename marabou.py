import numpy as np
from maraboupy import Marabou
import torch.nn as nn
import torch
from maraboupy import MarabouUtils
from maraboupy import Marabou, MarabouCore

import torch
import torch.nn as nn

# questa è la rete di partenza (quella che vogliamo verificare)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 2)
        self.output = nn.Linear(2, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# questa è un'altra rete che usa quella sopra in parallelo 2 volte (separatamente!)
class TwoCopyNet(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.net1 = base_net
        self.net2 = base_net

    def forward(self, x1, x2):
        y1 = self.net1(x1)
        y2 = self.net2(x2)
        return y1, y2


def formal_verification(network):
    tmp_onnx = 'tmp.onnx'
    network.eval()
    joint_model = TwoCopyNet(network)

    x1 = torch.zeros(1, 1)  
    x2 = torch.zeros(1, 1) 

    with torch.inference_mode():
        torch.onnx.export(joint_model,(x1, x2),tmp_onnx)

    Q = Marabou.read_onnx(tmp_onnx)

    # prima coppia input-output 
    X1 = Q.inputVars[0][0]
    X2 = Q.inputVars[1][0]

    # seconda coppia input-output
    Y1 = Q.outputVars[0][0]
    Y2 = Q.outputVars[1][0]

    # bounds prima coppia input-output
    Q.setLowerBound(X1[0], 0)
    Q.setUpperBound(X1[0], 1)

    # bounds seconda coppia input-output
    Q.setLowerBound(X2[0], 0)
    Q.setUpperBound(X2[0], 1)


    # x1 <= x2 (vincoli che legano variabili su coppie differenti)
    eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
    eq.addAddend(1, X1[0])
    eq.addAddend(-1, X2[0])
    eq.setScalar(0)
    Q.addEquation(eq)

    # y1 > y2 (vincoli che legano variabili su coppie differenti)
    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
    eq.addAddend(1, Y1[0])
    eq.addAddend(-1, Y2[0])
    eq.setScalar(0.00001) # (epsilon)
    Q.addEquation(eq)

    options = Marabou.createOptions(numWorkers=4,verbosity=0)
    sat, vals, stats = Q.solve(verbose=False, options=options)

    x1,y1 = None, None
    x2,y2 = None, None
    
    if sat == 'sat':
        # estrazione controesempio
        x1,y1 = [vals[v] for v in X1], [vals[v] for v in Y1]
        x2,y2 = [vals[v] for v in X2], [vals[v] for v in Y2]
        print("Marabou: x1={}, y1={}".format(x1,y1))
        print("Marabou: x2={}, y2={}".format(x2,y2))
    else:
        print("Marabou: no counterexample found")

    return (x1,y1), (x2,y2)

if __name__ == "__main__":
    network = SimpleNN()
    (x1,y1), (x2,y2) = formal_verification(network)
    
    if (x1,y1) != (None,None) and (x2,y2) != (None,None):
        print("check on the original NN")
        nn_x1 = torch.tensor(x1)
        nn_x2 = torch.tensor(x2)
        nn_y1 = network(nn_x1)
        nn_y2 = network(nn_x2)
        print("Original NN: x1={}, y1={}".format(nn_x1,nn_y1))
        print("Original NN: x2={}, y2={}".format(nn_x2,nn_y2))
        assert nn_x1 <= nn_x2 and not(nn_y1 <= nn_y2)