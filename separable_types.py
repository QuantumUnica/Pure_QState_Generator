import numpy as np
from functools import reduce

# Definizionni utili
identity_32x32 = np.eye(32)
identity_16x16 = np.eye(16)
identity_8x8 = np.eye(8)
identity_4x4 = np.eye(4)
identity_2x2 = np.eye(2)

swap_matrix = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])


# Separable 5 qubit
q5_sw12 = np.kron(swap_matrix, identity_8x8)                           # Swap_Id(8)      
q5_sw23 = np.kron(np.kron(identity_2x2, swap_matrix), identity_4x4)    # Id(2)_Swap_Id(4)
q5_sw34 = np.kron(np.kron(identity_4x4, swap_matrix), identity_2x2)    # Id(4)_Swap_Id(2)
q5_sw45 = np.kron(identity_8x8, swap_matrix)                           # Id(8)_Swap

qubits5_41_types = [ ("(AAAAb)", identity_32x32),
                    ("(AAAbA)", q5_sw45),
                    ("(AAbAA)",  np.dot(q5_sw34, q5_sw45)),
                    ("(AbAAA)", reduce(np.dot, [q5_sw23, q5_sw34, q5_sw45]) ),
                    ("(bAAAA)", reduce(np.dot, [q5_sw12, q5_sw23, q5_sw34, q5_sw45]) ) ]


qubits5_32_types = [ ("(AAABB)",  identity_32x32),
                    ("AABAB",  q5_sw34),
                    ("ABAAB",  reduce( np.dot, [q5_sw23, q5_sw34])),
                    ("BAAAB",  reduce( np.dot, [q5_sw12, q5_sw23, q5_sw34])),
                    ("BAABA",  reduce( np.dot, [q5_sw45, q5_sw12, q5_sw23, q5_sw34])),             
                    ("AABBA",  reduce( np.dot, [q5_sw45, q5_sw34])),     
                    ("ABABA",  reduce( np.dot, [q5_sw23, q5_sw45, q5_sw34]) ),
                    ("ABBAA",  reduce( np.dot, [q5_sw34, q5_sw23, q5_sw45, q5_sw34]) ),
                    ("BABAA",   reduce( np.dot, [q5_sw12, q5_sw34, q5_sw23, q5_sw45, q5_sw34]) ),
                    ("BBAAA",   reduce( np.dot, [q5_sw23, q5_sw12, q5_sw34, q5_sw23, q5_sw45, q5_sw34]) )     ]


qubits5_311_types = [ ("(AAAbb)",  identity_32x32),
                    ("(AAbAb)", q5_sw34),
                    ("(AAbbA)", reduce(np.dot, [q5_sw45, q5_sw34])),
                    ("(AbAAb)", reduce(np.dot, [q5_sw45, q5_sw23, q5_sw45, q5_sw34])),
                    ("(AbAbA)", reduce(np.dot, [q5_sw23, q5_sw45, q5_sw34])),
                    ("(AbbAA)", reduce(np.dot, [q5_sw34, q5_sw23, q5_sw45, q5_sw34])),
                    ("(bAbAA)", reduce(np.dot, [q5_sw12, q5_sw34, q5_sw23, q5_sw45, q5_sw34])),
                    ("(bbAAA)", reduce(np.dot, [q5_sw23, q5_sw12, q5_sw34, q5_sw23, q5_sw45, q5_sw34])),      
                    ("(bAAAb)", reduce(np.dot, [q5_sw12, q5_sw45, q5_sw23, q5_sw45, q5_sw34])),
                    ("(bAAbA)", reduce(np.dot, [q5_sw45, q5_sw12, q5_sw45, q5_sw23, q5_sw45, q5_sw34]))            ]


qubits5_221_types = [ ("AABBc",  identity_32x32),
                   ("ABABc",  q5_sw23),
                   ("BAABc",  reduce(np.dot, [q5_sw12, q5_sw23])),
                   ("AABcB",  q5_sw45),
                   ("ABAcB",  reduce(np.dot, [q5_sw23, q5_sw34, q5_sw45])),
                   ("BAAcB",  reduce(np.dot, [q5_sw12, q5_sw23, q5_sw34, q5_sw45])),
                   ("AAcBB",  reduce(np.dot, [q5_sw34, q5_sw45])),
                   ("ABcAB",  reduce(np.dot, [q5_sw34, q5_sw34, q5_sw45])),
                   ("ABcBA",  reduce(np.dot, [q5_sw45, q5_sw34, q5_sw34, q5_sw45])),
                   ("AcABB",  reduce(np.dot, [q5_sw23, q5_sw34, q5_sw45])),
                   ("AcBAB",  reduce(np.dot, [q5_sw34, q5_sw23, q5_sw34, q5_sw45])),
                   ("AcBBA",  reduce(np.dot, [q5_sw45, q5_sw34, q5_sw23, q5_sw34, q5_sw45])),
                   ("cAABB",  reduce(np.dot, [q5_sw12, q5_sw23, q5_sw34, q5_sw45])),
                   ("cABAB",  reduce(np.dot, [q5_sw34, q5_sw12, q5_sw23, q5_sw34, q5_sw45])),      
                   ("cABBA",  reduce(np.dot, [q5_sw45, q5_sw34, q5_sw12, q5_sw23, q5_sw34, q5_sw45])),      
                   ]
                   

qubits5_2111_types = [ ("AAbcd",  identity_32x32),
                    ("AbAcd", q5_sw12),
                    ("AbcAd", reduce(np.dot, [q5_sw34, q5_sw12])),
                    ("AbcdA", reduce(np.dot, [q5_sw45, q5_sw34, q5_sw12])),
                    ("bAAcd", reduce(np.dot, [q5_sw12, q5_sw12])),
                    ("bAcAd", reduce(np.dot, [q5_sw34, q5_sw12, q5_sw12])),
                    ("bcAAd", reduce(np.dot, [q5_sw23, q5_sw34, q5_sw12, q5_sw12])),
                    ("bAcdA", reduce(np.dot, [q5_sw12, q5_sw45, q5_sw34, q5_sw12])),
                    ("bcAdA", reduce(np.dot, [q5_sw23, q5_sw12, q5_sw45, q5_sw34, q5_sw12])),
                    ("bcdAA", reduce(np.dot, [q5_sw34, q5_sw23, q5_sw12, q5_sw45, q5_sw34, q5_sw12])),        ]

separable_types_5qubit = [("Qubits Gropuing [4|1] (AAAA|b)", [16,2], qubits5_41_types), 
                       ("Qubits Gropuing [3|2]  (AAA|BB)", [8, 4], qubits5_32_types), 
                       ("Qubits Gropuing [3|1|1]  (AAA|b|c)", [8, 2, 2], qubits5_311_types),
                       ("Qubits Gropuing [2|2|1]  (AA|BB|c)", [4, 4, 2], qubits5_221_types),
                       ("Qubits Gropuing [2|1|1|1]  (AA|b|c|d|)", [4, 2, 2, 2], qubits5_2111_types)]


# Separable 4 qubit
q4_sw12 = np.kron(swap_matrix, identity_4x4)                           # Swap_Id(4)      
q4_sw23 = np.kron(np.kron(identity_2x2, swap_matrix), identity_2x2)    # Id(2)_Swap_Id(2)
q4_sw34 = np.kron(identity_4x4, swap_matrix)                           # Id(4)_Swap



qubits4_31_types = [ ("AAAb", identity_16x16),
                    ("AAbA", q4_sw34),
                    ("AbAA",  np.dot(q4_sw23, q4_sw34)),
                    ("bAAA", np.dot( np.dot( q4_sw12, q4_sw23), q4_sw34) ) ]


qubits4_22_types = [ ("AABB",  identity_16x16),
                    ("ABAB",  q4_sw23),
                    ("ABBA", np.dot(q4_sw34, q4_sw23)) ]


qubits4_211_types = [ ("AAbc",  identity_16x16),
                    ("AbAc", q4_sw23),
                    ("AbcA", np.dot(q4_sw34, q4_sw23)),
                    ("bAcA", np.dot(np.dot(q4_sw12, q4_sw34), q4_sw23)),
                    ("bcAA", reduce(np.dot, [q4_sw23, q4_sw12, q4_sw34, q4_sw23])),
                    ("bAAc", reduce(np.dot, [q4_sw34, q4_sw12, q4_sw34, q4_sw23]))     ]
                    

separable_types_4qubit = [("Qubits Gropuing [3|1] (AAA|b)", [8,2], qubits4_31_types), 
                       ("Qubits Gropuing [2|2]  (AA|BB)", [4, 4], qubits4_22_types), 
                       ("Qubits Gropuing [2|1|1]  (AA|b|c)", [4, 2, 2], qubits4_211_types)]



# Separable 3 qubit
q3_sw12 = np.kron(swap_matrix, identity_2x2)
q3_sw23 = np.kron(identity_2x2, swap_matrix)  

qubits3_21_types = [("AAb",  np.eye(8)),
                    ("AbA", q3_sw23),
                    ("bAA", np.dot(q3_sw12, q3_sw23))      ]

separable_types_3qubit = [("Qubits Gropuing [2|1] (AA|b)", [4,2], qubits3_21_types)]