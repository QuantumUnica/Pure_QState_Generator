import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

N_qubit = 2
ineq_type = 'chsh'
dataset_fName = f'dati_paper/200000_EFS_{ineq_type}_States_2Classes_2_qubit_float64.csv'
#dataset_fName = f'400000_EFS_{ineq_type}_States_3Classes_3_qubits_float64.csv'

nQubits = dataset_fName.split('_')[-3]
nEFS_Classes = int([x[0] for x in dataset_fName.split('_') if 'Classes' in x][0])

sizes = [1000]

test_size = 0.2

df = pd.read_csv(dataset_fName, header=None)

for s in sizes:
    nSamp = int(np.ceil(np.ceil(s/2)/nEFS_Classes))

    nonVio_SepStates, vio_SepStates = [], []
    
    
    #df = df[df.iloc[:, -2] == df.iloc[:, -4]] # Take only states whose locality is violated for both ineq.
   
    d = df.to_numpy()
    nonVioIds = set(np.where(d[:,-2]==0)[0])
    vioIds = set(np.where(d[:,-2]==1)[0])

    fatIds = set(np.where(d[:,-3]==1)[0])
    sepIds = set(np.where(d[:,-3]==2)[0])
    entIds = set(np.where(d[:,-3]==3)[0])
    nonVio_fatIds = list(set.intersection(nonVioIds, fatIds))
    nonVio_sepIds = list(set.intersection(nonVioIds, sepIds))
    nonVio_entIds = list(set.intersection(nonVioIds, entIds))

    print('\nNon Vio states \n ---------------------------------------')
    print(f" Ent {len(list(nonVio_entIds))}\n Fact {len(list(nonVio_fatIds))}\n Sep {len(list(nonVio_sepIds))}")

    d_nonVioFat = d[nonVio_fatIds, ...]
    d_nonVSep = d[nonVio_sepIds, ...]
    d_nonVEnt = d[nonVio_entIds, ...]

    rng = np.random.default_rng()
    
    nonVio_FatStates = rng.choice(d_nonVioFat, nSamp)
    nonVio_EntStates = rng.choice(d_nonVEnt, nSamp)
    if len(sepIds): 
        nonVio_SepStates = rng.choice(d_nonVSep, nSamp)
    
    vio_fatIds = list(set.intersection(vioIds, fatIds))
    vio_sepIds = list(set.intersection(vioIds, sepIds))
    vio_entIds = list(set.intersection(vioIds, entIds))

    print('\nVio states \n ---------------------------------------')
    print(f" Ent {len(list(vio_entIds))}\n Fact {len(list(vio_fatIds))}\n Sep {len(list(vio_sepIds))}")

    d_vioFat = d[vio_fatIds, ...]
    d_vioSep = d[vio_sepIds, ...]
    d_vioEnt = d[vio_entIds, ...]

    vio_FatStates = rng.choice(d_vioFat, nSamp)
    vio_EntStates = rng.choice(d_vioEnt, nSamp)
    if len(sepIds): 
        vio_SepStates = rng.choice(d_vioSep, nSamp)

    if len(sepIds):
        vio = np.concatenate([vio_FatStates, vio_SepStates, vio_EntStates], axis=0)
        nonVio = np.concatenate([nonVio_FatStates, nonVio_SepStates, nonVio_EntStates], axis=0)
    else:
        vio = np.concatenate([vio_FatStates, vio_EntStates], axis=0)
        nonVio = np.concatenate([nonVio_FatStates, nonVio_EntStates], axis=0)

    print("\n", vio.shape, nonVio.shape)

    X = np.concatenate([vio, nonVio], axis=0)
    yNonLoc = X[..., -2].astype(int)
    y_EFS = X[:, -3].astype(int)
    
    X = X[..., :-3]

    XDf = pd.DataFrame(X)
    yDfNonLoc = pd.DataFrame(yNonLoc)
    yDfEFS = pd.DataFrame(y_EFS)
        
    dfTemp = pd.concat([XDf, yDfNonLoc, yDfEFS], axis=1)
    dfTemp.columns = range(len(dfTemp.columns))
    assert len(dfTemp) >= s, "Not enough states"

    # Remove states used for in the current subset
    columns_to_check = list(range((2**N_qubit)*2))
    common_rows = pd.merge(df[columns_to_check], dfTemp[columns_to_check], how='inner')
    df = df[~df[columns_to_check].apply(tuple, axis=1).isin(common_rows.apply(tuple, axis=1))]
    
    dfTrain, dfTest = train_test_split(dfTemp, test_size=0.2, stratify=dfTemp.iloc[:,-2:], random_state=42)

    dfTrain.columns = list(range(0, len(dfTrain.columns)))
    dfTest.columns = list(range(0, len(dfTest.columns)))

    dfTrain.drop(dfTrain.columns[-1], axis=1).to_csv('{}_EFS_States_{}_{}_qubit_float64_Train.csv'.format(s, ineq_type, nQubits) ,index=False, header=False)
    dfTest.drop(dfTest.columns[-1], axis=1).to_csv('{}_EFS_States_{}_{}_qubit_float64_Test.csv'.format(s, ineq_type, nQubits) ,index=False, header=False)

    dfTrain.drop(dfTrain.columns[-2], axis=1).to_csv('{}_EFS_States_{}_qubit_float64_Train.csv'.format(s, nQubits) ,index=False, header=False)
    dfTest.drop(dfTest.columns[-2], axis=1).to_csv('{}_EFS_States_{}_qubit_float64_Test.csv'.format(s, nQubits) ,index=False, header=False)
