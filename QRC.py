import random 
import math as m
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix


############################################
##### Quantum Random Circuit Generator #####
############################################


def QRC_Marmot(N, D, max_gates):
    '''
    Generates a random quantum circuit for the AQT Marmot device.
    Args:
        N (int): Number of qubits.
        D (int): Integer representing the circuit depth.
        max_gates (int): Maximum number of gates applied per iteration (1 or 2-qubit gates).
    Returns:
        dict: A dictionary containing the ideal circuit and its density matrix.'''
    
    # define gate names
    Gate_Names_1q = ['r', 'rz']  # One-qubit gates
    Gate_Names_2q = ['rxx']   # Two-qubit gates

    # if depth is not specified, choose a random depth
    if D is None:
        D = random.randint(10, 50)
    
    # initialize the circuit
    circ = QuantumCircuit(N)    

    gate_counts = {
        "r": 0,
        "rz": 0,
        "rxx": 0
    }
    
    # iterate through the layers up to specified circuit depth D
    for _ in range(D):
        # L is the list to hold the operations for each layer
        L = []
        r = range(N)
        qubits_numbers = list(r)
        choices = []
        
        # create choices based on max_gates
        if max_gates == 3:
            while 2 < len(qubits_numbers):
                a = random.randint(1, 3)
                b = random.sample(qubits_numbers, k=a)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
            while 1 < len(qubits_numbers):
                a = random.randint(1, 2)
                b = random.sample(qubits_numbers, k=a)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
            if 0 < len(qubits_numbers):
                choices = choices + [[qubits_numbers[0]]]
        elif max_gates == 2:
            while 1 < len(qubits_numbers):
                a = random.randint(1, 2)
                b = random.sample(qubits_numbers, k=a)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
            if 0 < len(qubits_numbers):
                choices = choices + [[qubits_numbers[0]]]
        elif max_gates == 1:
            while 0 < len(qubits_numbers):
                b = random.sample(qubits_numbers, k=1)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)                
        # generate corresponding random gates
        for x in choices:
            if len(x) == 1:
                a = "."
                b = random.choice(Gate_Names_1q)
                gate_counts[b] += 1
                if b == "r":
                # apply r gate with random angles
                    angle1 = str(2 * (m.pi) * (random.random()))
                    angle2 = str(2 * (m.pi) * (random.random()))
                    qubit1 = str(x[0])
                    c =  a + b + "(" + angle1 + "," + angle2 + "," + qubit1 + ")" 
                if b == "rz":
                # Apply rz gate with random angle
                    angle1 = str(2 * (m.pi) * (random.random()))
                    qubit1 = str(x[0])
                    c =  a + b + "(" + angle1 + "," + qubit1 + ")"     
                L.append(c)
            elif len(x) == 2:
                # apply rxx gate
                a = "."
                b = random.choice(Gate_Names_2q)
                gate_counts[b] += 1
                if b == 'rxx':
                    # add both directions
                    angle1 = str( (1/2) * (m.pi) * (random.random()))
                    qubit1 = str(x[0])
                    qubit2 = str(x[1])
                    c = a + b + "(" + angle1 + "," + qubit1 + "," + qubit2 + ")"
                L.append(c)
        
        # add all instructions to the circuit
        for instr in L:
            eval("circ" + instr) 
    
    # get density matrix of the ideal circuit
    density_matrix = DensityMatrix(circ.reverse_bits()).data
    
    
    return {
        "Ideal Circuit": circ,
        "Ideal RHO": density_matrix,
    }

