import os 
import json

PROPERTIES = [1,2,3,4,14]
BRAINS = [(10,6)]
LEARNING_RATES = [1e-2,1e-3,1e-4]
SEEDS = [100,200,300]

STRATEGIES = [(25,20),(50,10),(50,20),(50,30),(75,20)]

SOFT = ['','-SA']
EPSILON = [1e-4]

key_template = (
    '{property}-'
    '{epsilon}-'
    '{learning_rate}-'
    '{layers}x{neurons}-'
    '{counterexamples}-'
    '{sampling}-'
    '{seed}-L'
    '{soft}'
    )

KEYS = [key_template.format(
    property=p, 
    epsilon=epsilon, 
    learning_rate=learning_rate, 
    layers=layers,
    neurons=neurons,
    counterexamples=counterexamples, 
    sampling=sampling, 
    seed=seed, 
    soft=soft
    ) \
    for p in PROPERTIES \
    for epsilon in EPSILON \
    for learning_rate in LEARNING_RATES \
    for (layers,neurons) in BRAINS \
    for (counterexamples,sampling) in STRATEGIES \
    for seed in SEEDS \
    for soft in SOFT]


def load_experiment_data():
    data = dict()
    for key in KEYS:
        path = 'experiments/{}'.format(key)
        if os.path.exists(path):
            with open('{}/stats.json'.format(path), "r") as f:
                data[key] = json.load(f)
    return data

if __name__ == '__main__':
    data = load_experiment_data()
    total = len(data)
    missing = len(set(KEYS) - data.keys())
    timeout = sum(1 for k in data.keys() if data[k]['TotalTIME'] == None)

    print('total={}'.format(total))
    print('missing={}'.format(missing))
    print('timeout={}'.format(timeout))