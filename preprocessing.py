import pandas as pd
import conf
import itertools

def ViolationsToP1(df):
    pre  = df['Distance to closest child'] <= 0.1
    post = df['Mopping the floor'] == df.filter(items=list(conf.OutputScores.keys())).min(axis=1)
    mask = pre & (~post)
    return df.loc[mask]

def ViolationsToP2(df):
    dfH = df[df['Distance to closest human']  <= 0.1].copy()
    dfA = df[df['Distance to closest animal'] <= 0.1].copy()
    dfH['_tmp'] = 1
    dfA['_tmp'] = 1
    cross = pd.merge(dfH, dfA, on='_tmp', suffixes=('_H', '_A')).drop('_tmp', axis=1)
    mask = cross['Mopping the floor_H'] > cross['Mopping the floor_A']
    return cross[mask]


def ViolationsToP3(df):
    dfA = df[df['Distance to closest animal'] <= 0.1].copy()
    dfH = df[df['Distance to closest human']  <= 0.1].copy()
    dfA['_tmp'] = 1
    dfH['_tmp'] = 1
    cross = pd.merge(dfA, dfH, on='_tmp', suffixes=('_A', '_H')).drop('_tmp', axis=1)
    mask = cross['Mopping the floor_A'] > cross['Mopping the floor_H']
    return cross[mask]


def ViolationsToP4(df):
    pre  = df['Distance to closest child'] >= 0.75
    post = (df['Carry warm food'] == df.filter(items=list(conf.OutputScores.keys())).max(axis=1)) | (df['Carry big objects (tables, chairs)'] == df.filter(items=list(conf.OutputScores.keys())).max(axis=1))
    mask = pre & (~post)
    return df.loc[mask]

df = pd.read_csv("not_normallized.csv").iloc[:,1:]
df.iloc[:,1] = 1 - df.iloc[:,1]
df = df.iloc[:,1:]
df.rename(columns={"Using arrow": "mode"}, inplace=True)

MAX = {k : x[-1] for (k,x) in conf.InputFeatures.items()} | {k : y[-1] for (k,y) in conf.OutputScores.items()}

df = df.div(MAX)
df = df.clip(0,1)

df.to_csv('baseline.csv', index=False, header=True)

#print(df.min())
#print(df.max())

for f in [ViolationsToP1, ViolationsToP2, ViolationsToP3, ViolationsToP4]:
	BadRows = f(df)
	print('{}={}'.format(f.__name__, len(BadRows)))

	#df.to_csv('normalized.csv', index=False, header=False)


