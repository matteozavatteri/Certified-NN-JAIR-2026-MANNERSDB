import os 

import experiments
import statistics
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

data = experiments.load_experiment_data()

with open('table-sampling.tex', 'w') as f:
    f.write('\\scalebox{0.55}{\\begin{tabular}{|*{16}{c|}}\n')
    f.write('\\hline\n')
    
    f.write(
        ('\\multicolumn{3}{|l|}{} & '
        '\\multicolumn{13}{l|}{{\\bf Average value $\\pm$ standard deviation}} \\\\\n'))    
    
    f.write('\\cline{4-16}\n')
    
    f.write(
            ('\\multicolumn{3}{|l|}{} &'
             '\\multicolumn{2}{l|}{{\\bf Generic synthesis process}} &'
             '\\multicolumn{11}{l|}{{\\bf Generic CEGIS iteration}} \\\\\n')
    )
    
    f.write('\\hline\n')
    
    f.write(
        (' & '
        '{\\bf \\makecell{MaxS}} & '
        '{\\bf \\makecell{TO}} & '
        '{\\bf \\makecell{CEGIS\\\\iterations}} & '
        '{\\bf \\makecell{Total\\\\time (s)}} & '
        '{\\bf \\makecell{Training\\\\time (s)}} & '
        '{\\bf \\makecell{Opportun. \\\\search time (s)}} & '
        '{\\bf \\makecell{FV \\\\ time (s)}} & '
        '{\\bf \\makecell{Repair \\\\ time (s)}} & '
        '{\\bf \\makecell{Initial loss \\\\ on valid}} & '
        '{\\bf \\makecell{Final loss \\\\ on valid}} & '
        '{\\bf \\makecell{Initial loss \\\\ on test}} & '
        '{\\bf \\makecell{Final loss \\\\ on test}} & '
        '{\\bf \\makecell{CounterEX \\\\ (opportun.)}} & '
        '{\\bf \\makecell{FV \\\\ calls}} & '
        '{\\bf \\makecell{New \\\\ pairs}}  \\\\\n'))
    f.write('\\hline\n')

    for P in experiments.PROPERTIES:
        f.write("\\multirow{{{}}}{{*}}{{$P_{{{}}}$}} ".format(2*3,P))
        for sampling in [10,20,30]:
            first = True
            if first:
               f.write("& \\multirow{{{}}}{{*}}{{${{{}}}$}} ".format(2,sampling))
            for soft in experiments.SOFT:
                ALL = [experiments.key_template.format(
                    property=P, 
                    epsilon=epsilon, 
                    learning_rate=learning_rate, 
                    layers=10,
                    neurons=6,
                    counterexamples=50, 
                    sampling=sampling, 
                    seed=seed, 
                    soft=soft
                    ) \
                for epsilon in experiments.EPSILON \
                for learning_rate in experiments.LEARNING_RATES \
                for seed in experiments.SEEDS]

                ROW = dict()

                # remove all experiments that hit the timeout
                TERMINATED = [k for k in ALL if data[k]['TotalTIME'] is not None]

                TO = len(ALL) - len(TERMINATED)
                ROW['CEGISiterations'] = '${}\\pm{}$'.format(
                        round(statistics.mean(data[k]['CEGISiterations'] for k in TERMINATED)), \
                        round(statistics.stdev(data[k]['CEGISiterations'] for k in TERMINATED)) \
                    )

                
                ROW['TotalTIME'] = '${}\\pm{}$'.format(
                        round(statistics.mean(data[k]['TotalTIME'] for k in TERMINATED) ,2), \
                        round(statistics.stdev(data[k]['TotalTIME'] for k in TERMINATED) ,2) \
                    )
                
                for metric in ['TrainingTIME', 'OpportunisticSeachTIME', 'FormalVerificationTIME', 'RepairTIME']:
                    tmp = list(data[k][metric] for k in TERMINATED)
                    tmp = [x for y in tmp for x in y]
                    ROW[metric]  = '${}\\pm{}$'.format(
                        round(statistics.mean(tmp),2), \
                        round(statistics.stdev(tmp),2) \
                    )

                for metric in ['OpportunisticCEX', 'NewPairs']:
                    tmp = list(data[k][metric] for k in TERMINATED)
                    tmp = [x for y in tmp for x in y]
                    ROW[metric]  = '${}\\pm{}$'.format(
                        round(statistics.mean(tmp)), \
                        round(statistics.stdev(tmp)) \
                    )

                tmp = [data[k]['FormalVerificationCEX'].count(1) for k in TERMINATED]
                ROW['CallsToFV']  = '${}\\pm{}$'.format(
                    round(statistics.mean(tmp)), \
                    round(statistics.stdev(tmp)) \
                )

                for metric in ['LossOnValidation', 'LossOnTest']:
                    ROW['Initial' + metric] = '${}\\pm{}$'.format(
                        round(statistics.mean(data[k][metric][0]  for k in TERMINATED),3), \
                        round(statistics.stdev(data[k][metric][0]  for k in TERMINATED),3) \
                    )

                    ROW['Final' + metric] = '${}\\pm{}$'.format(
                        round(statistics.mean(data[k][metric][-1]  for k in TERMINATED),3), \
                        round(statistics.stdev(data[k][metric][-1]  for k in TERMINATED),3),
                    )

                COLOR = '\\cellcolor{lightgray!50}' if soft == '-SA' else ''

                if not first:
                    f.write(' & ')
                first = False
                f.write(
                    (' & '
                   '{COLOR}{TO} & '
                    '{COLOR}{CEGISiterations} & '
                    '{COLOR}{TotalTIME} & '
                    '{COLOR}{TrainingTIME} & '
                    '{COLOR}{OpportunisticSeachTIME} & ' 
                    '{COLOR}{FormalVerificationTIME} & ' 
                    '{COLOR}{RepairTIME} & ' 
                    '{COLOR}{InitialLossOnValidation} & ' 
                    '{COLOR}{FinalLossOnValidation} & ' 
                    '{COLOR}{InitialLossOnTest} & ' 
                    '{COLOR}{FinalLossOnTest} & ' 
                    '{COLOR}{OpportunisticCEX} & ' 
                    '{COLOR}{CallsToFV} & ' 
                    '{COLOR}{NewPairs} \\\\\n'.format(
                        COLOR=COLOR,
                        TO=TO,
                        CEGISiterations=ROW['CEGISiterations'],
                        TotalTIME=ROW['TotalTIME'],
                        TrainingTIME=ROW['TrainingTIME'],
                        OpportunisticSeachTIME=ROW['OpportunisticSeachTIME'],
                        FormalVerificationTIME=ROW['FormalVerificationTIME'],
                        RepairTIME=ROW['RepairTIME'],
                        InitialLossOnValidation=ROW['InitialLossOnValidation'],
                        FinalLossOnValidation=ROW['FinalLossOnValidation'],
                        InitialLossOnTest=ROW['InitialLossOnTest'],
                        FinalLossOnTest=ROW['FinalLossOnTest'],
                        OpportunisticCEX=ROW['OpportunisticCEX'],
                        CallsToFV=ROW['CallsToFV'],
                        NewPairs=ROW['NewPairs'])
                    )
                )
            if sampling != 30:
                f.write('\\cline{2-16}\n')
        f.write('\\hline\n')

    f.write('\n\\end{tabular}}')
