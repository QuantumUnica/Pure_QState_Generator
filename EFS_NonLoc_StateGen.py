#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:37:02 2023

@author: carlo
"""

import numpy as np
import pandas as pd
from functools import reduce
from itertools import product
import random
from qiskit.quantum_info import Statevector
from separable_types import separable_types_3qubit, separable_types_4qubit, separable_types_5qubit
from collections import Counter
import math
from tqdm import tqdm

import Inequalities


# converte un operatore di densità complesso in un vettore reale
def to_real_vector(complex_matrix):
    real_part = np.real(complex_matrix)
    imag_part = np.imag(complex_matrix)
    res = []
    for r, i in zip(real_part, imag_part):
        res.append(r)
        res.append(i)
    return np.array(res)

# Cast every row of a the input matrix into complex vector
def to_complex_vector(x):
    n_floatFeatures = x.shape[1]
   
    even_idx = [x for x in range(n_floatFeatures) if x%2==0]
    odd_idx = [x for x in range(n_floatFeatures) if x%2!=0]

    real = x[:,even_idx]
    imag = x[:,odd_idx]

    return np.vectorize(complex)(real, imag)

# Esegue il check di uno state vector attraverso quiskit
def check_state(qu_state_vector):
    v = Statevector(qu_state_vector)
    if v.is_valid(): 
        return 1
    else:
        print('\nNumpy array in data attribute: \n')
        print(v.data)
        print( 'Num qubit: ', v.num_qubits, 
            '\nPurity: ', v.purity(), 
            '\nIs a valid quantum state? ', 'Yes.' if v.is_valid() else 'No.')

        print(f"Obj. Type {type(v.data)}")
        print(f"Dtype {v.data.dtype}")
        print(f"Shape {v.data.shape}")
        print(f"N.Dim {v.data.ndim}")
        return 0


def check_locality(state, ineq='svet', n_shots=5):
    state = to_complex_vector(np.expand_dims(state, axis=0)) 
    
    row_vector = state.reshape((state.shape[0],state.shape[1],1))
    col_vector = state.reshape((state.shape[0],1,state.shape[1]))
    rho = row_vector @ col_vector
    rho = rho[-1,...]

    violation_degree = Inequalities.test_ineq_on_state(rho, ineq, seeds=n_shots) 
    return violation_degree
    
    


# Crea casualmente uno stato di un sistema composto da "size" qubits
def generate_random_state(size):
    real_part = np.random.rand(size) * 2 - 1  # Parte reale compresa tra -1 e 1
    imag_part = np.random.rand(size) * 2 - 1  # Parte immaginaria compresa tra -1 e 1
    s = real_part + 1j * imag_part

    #s = random_statevector((size)) 
    s /= np.linalg.norm(s)
    return s



def factorized(num_qubits):     
    tensor_product = reduce(np.kron, [generate_random_state(2) for _ in range(num_qubits)])
    assert check_state(tensor_product), 'Not valid quantum state'
    return tensor_product


def entangled(num_qubits):
    state = generate_random_state(2**num_qubits)
    assert check_state(state), 'Not valid quantum state'
    return state


# Il numero di qubits è determinato esternamente tramite il sepType
def separable(sepType):
    initial_random_states = []  # conterrà gli stati necessari alla creazione dello stato separable iniziale
    tensor_product = None       # lo stato iniziale su cui sarà applicato l'oeratore
    chosen_sep_type = None      # stringa descrittiva del tipo di separable
    subtypes = None             # sottotipi dello stato separable selezionato
    str_chosen_operator = None      # operatore che permette di ottenere un dato sottotipo di separable

    # ty è una tripla, quindi posso indicizzarla per selezionare:
    chosen_sep_type = sepType[0]        # la stringa relativa al type selezionato random
    initial_random_states = [ generate_random_state(length) for length in ty[1]]  # la lunghezza degli stati iniziali
    subtypes = sepType[2]     # i sottotipi del tipo di separable selezionato casualmente

    tensor_product = reduce(np.kron, initial_random_states)   # creo lo stato separable iniziale

    # Trasforma lo stato Separable in uno dei suoi sottotipi 
    random_operator = random.choice(subtypes)      # Seleziona casualmente il sottotipo di separable
    str_chosen_operator = random_operator[0]           # Salva la stringa che descrive l'operatore relezionato
    state = random_operator[1] @ tensor_product     # Seleziona l'operatore e lo applica allo stato iniziale 
    assert check_state(state)
    return state, chosen_sep_type    # stato finale prodotto dall'operatore e label

   

if __name__ == '__main__':

    system_sizes = list(range(2,6))
    data_sizes = [10000]
    ineq_to_check = []#['svet', 'mer', 'chsh']
    
    assert all([1 for x in ineq_to_check if x in ['svet', 'mer', 'chsh']])
    
    for n_qubits, dataset_total_objects in list(product(system_sizes,data_sizes)):
        
        if n_qubits > 2:
            ineq_to_check = [x for x in ineq_to_check if x != 'chsh']
        
        thresh = Inequalities.get_loc_violation_thresholds(n_qubits)

        # Corrispondenza tra label stringa e integer 
        if n_qubits == 3:
            l_enc = {'[2|1]': 'Sep'}
        elif n_qubits == 4:
            l_enc = { k:f'Sep{i+1}' for i,k in enumerate(['[3|1]', '[2|2]', '[2|1|1]'])}
        elif n_qubits == 5:
            l_enc = { k:f'Sep{i+1}' for i,k in enumerate(['[4|1]','[3|2]', '[3|1|1]', '[2|2|1]', '[2|1|1|1]'])}
             

        if n_qubits > 2: m_sep_types = len(l_enc.items())    # Numero di classi separable
        else: m_sep_types = 0

        n_classes = m_sep_types + 2         # Numero totale di classi
        N = math.ceil(dataset_total_objects/n_classes)    # Numero di oggetti per classe 

        factorized_ds = [ factorized(n_qubits) for _ in range(N) ]
        entangled_ds = [ entangled( n_qubits) for _ in range(N) ]
        
        data = [factorized_ds, entangled_ds]
        labels = [ np.array(['Fact' for i in range(len(factorized_ds))]),                               
                            np.array(['Ent' for i in range(len(entangled_ds))])]
        
        # Separble state generation
        if n_qubits > 2:
            sep_states = []
            sep_labels = []

            if n_qubits == 3:
                separable_types = separable_types_3qubit 
            elif n_qubits==4: 
                separable_types = separable_types_4qubit
            elif n_qubits==5:
                separable_types = separable_types_5qubit
            
            # NOTE
            samples_per_type = N#math.ceil(N/len(separable_types) )
            
            # separable_types è una lista di triple in cui ogni elemento 
            # mantiene la stringa che descrive la tipologia di separable, 
            # i vettori che descrivono lo stato separable iniziale,
            # e un dizionario che descrive i sottotipi del tipo di separable
            for ty in separable_types:
                for _ in range(samples_per_type):           
                    s, lab = separable(ty) # creazione dello stato
                    sep_states.append(s)
                    sep_labels.append(lab)

            data.append( sep_states )
            labels.append( [l_enc[x.split(' ')[2]] for x in sep_labels] )
                    
        X = np.concatenate(data, axis=0)
        y = np.concatenate(labels, axis=0)

        print('-'*30, '\n class label, \t n_samples \n', '-'*30)
        for l, c in Counter(y).items():
            print(l, '\t\t' ,c)

        X_ = np.zeros((X.shape[0], X.shape[1]*2), dtype=np.float64)
        for i in range(len(X)):
            X_[i] = to_real_vector(X[i])
            #X_[i] = X[i]
        X = X_
        
        # Crea dataframe per dati e labels
        XDf = pd.DataFrame(X)
        yDf = pd.DataFrame(y) 
        dfColList = [XDf, yDf]
        col_names = list(range(X.shape[1]))+['state_type']

        for ineq_type_str in ineq_to_check:

            locality_values = []
            for i in tqdm(range(X.shape[0]), miniters=10):
                v = check_locality(X[i], ineq_type_str)
                locality_values.append(v)

            locality_labels = np.array([1 if value > thresh[ineq_type_str] else 0 for value in locality_values])
            
            locLabDf = pd.DataFrame(locality_labels)
            locValDf = pd.DataFrame(locality_values)

            dfColList+=[locLabDf, locValDf]
            col_names+=[ineq_type_str+'_non_local', ineq_type_str+'_value']
        
        out_df = pd.concat(dfColList, axis=1)
        out_df.columns = col_names

        if N*n_classes > dataset_total_objects:
            column_name = 'state_type'
            target_string = 'Ent'
            excess = N*n_classes - dataset_total_objects

            rows_to_remove = out_df[out_df[column_name] == target_string].head(excess)
            out_df = out_df.drop(rows_to_remove.index)
        
        out_df.to_csv('{}_EFS_States_{}Classes_{}_qubit_NonLoc_{}.csv'.format(len(out_df), n_classes, n_qubits, X.dtype), 
                                            index=False, header=False)
                

        print('Total number of generated samples', len(y))
        print('Total number of samples in output file', len(out_df))

    print()