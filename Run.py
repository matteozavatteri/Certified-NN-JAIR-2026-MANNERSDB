import subprocess

# in seconds
TIMEOUT=3600 * 5

P = [1,2,3,4,14]
E = 100 # epochs
S = 20  # early stopping

# layers x neurons
H = [(10,6)]

# learning rate
LR = [0.01, 0.001, 0.0001]

# seeds
SEEDS = [100,200,300]

# strategy (counterexamples, sampling)
STRATEGY = [(25,20),(50,10),(50,20),(50,30),(75,20)]

KEEP = ['','--keep']
KEEP = ['']

SOFT = ['','--soft-acceleration']
# command
c = (
    'python RobotCertifier.py {property} '
    '--learning-rate {lr} '
    '--epochs {epochs} '
    '--early-stopping {earlystopping} '
    '--hidden-layers {layers} '
    '--neurons-per-hidden-layer {neurons} '
    '--counterexamples {counterexamples} '
    '--sampling {sampling} '
    '--seed {seed} {sa} '
    '--cores 20 {keep}'
    )

cmds = [c.format(
    property=p, 
    lr=lr, 
    epochs=E, 
    earlystopping=S, 
    layers=layers, 
    neurons=neurons, 
    counterexamples=ce, 
    sampling=sam, 
    seed=seed, 
    sa=sa, 
    keep=keep) for p in P for lr in LR for (layers,neurons) in H for seed in SEEDS for (ce,sam) in STRATEGY for sa in SOFT for keep in KEEP]

for cmd in cmds:
    #print("Running: {}".format(cmd))
    try:
        p = subprocess.run(cmd.split(), stdout=subprocess.PIPE, timeout=TIMEOUT)
    except subprocess.CalledProcessError as e:
        print("{}".format(e))
        print("going ahead")
    except:
        print('timeout\n')

