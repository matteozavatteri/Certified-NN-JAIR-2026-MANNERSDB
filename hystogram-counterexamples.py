import experiments
import statistics
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

data = experiments.load_experiment_data()
plot_rows = []

for P in experiments.PROPERTIES:
    fv_values = []
    repair_values = []

    ALL = [experiments.key_template.format(
            property=P,
            epsilon=epsilon,
            learning_rate=learning_rate,
            layers=10,
            neurons=6,
            counterexamples=counterexamples,
            sampling=sampling,
            seed=seed,
            soft=soft
        )
        for epsilon in experiments.EPSILON
        for learning_rate in experiments.LEARNING_RATES
        for counterexamples in [25, 50, 75]
        for sampling in [20]
        for seed in experiments.SEEDS
        for soft in experiments.SOFT
        ]

    TERMINATED = [k for k in ALL if data[k]['TotalTIME'] is not None]
    tmp = list(data[k]['FormalVerificationTIME'] for k in TERMINATED)
    tmp = [x for y in tmp for x in y]
    FV = round(statistics.mean(tmp),2)

    tmp = list(data[k]['RepairTIME'] for k in TERMINATED)
    tmp = [x for y in tmp for x in y]
    R = round(statistics.mean(tmp),2)
    
    plot_rows.append({
        'P': P,
        'FV': FV,
        'R': R,
    })

labels = ['$P_{}$'.format(r['P']) for r in plot_rows]
labels[-1] = '$\\mathcal{P}$'
fv_means = [r['FV'] for r in plot_rows]
repair_means = [r['R'] for r in plot_rows]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))

ax.bar(x - width / 2, fv_means, width, label='Formal Verification time')
ax.bar(x + width / 2, repair_means, width, label='Repair time')

ax.set_ylabel('Time (s)')
ax.set_xlabel('Property')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('img/fv-vs-rep-robot-counterexamples.pdf')
plt.close()