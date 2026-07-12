import numpy as np 
from qiskit import QuantumCircuit
import itertools
import os

def theoretical_Rz(theta):
    R = np.array([[np.exp(-1j * theta/2) , 0], [0 , np.exp(1j * theta/2)]])
    return R

def theoretical_Rx(theta):
    R = np.array([[np.cos(theta/2), (-1j)*np.sin(theta/2)],
        [(-1j)*np.sin(theta/2), np.cos(theta / 2)]])
    return R

def theoretical_R(theta, phi):
    cos_theta_over_2 = np.cos(theta / 2)
    sin_theta_over_2 = np.sin(theta / 2)
    R = np.array([
        [cos_theta_over_2, -1j * np.exp(-1j * phi) * sin_theta_over_2],
        [-1j * np.exp(1j * phi) * sin_theta_over_2, cos_theta_over_2]
    ])
    return R

def R_sigma_z_Rdagger(theta, phi):
    sigma_z = np.array([[1, 0], [0, -1]])
    R = theoretical_R(theta, phi)
    R_dagger = R.conj().T
    return R @ sigma_z @ R_dagger

# Computing R_dagger * sigma_z * R
def R_dagger_sigma_z_R(theta, phi):
    sigma_z = np.array([[1, 0], [0, -1]])
    R = theoretical_R(theta, phi)
    R_dagger = R.conj().T
    return R_dagger @ sigma_z @ R

############################################################
### Svetlichny inequality functions ########################
############################################################
def nu(k, branch = False):
    nu0 = (-1)**(k*(k+1)/2)
    return nu0

def sign(t):
    s = t.count(1)
    return s


def KroneckerTwoSets(L1, L2):
    P = []
    for i in range(len(L1)):
        for j in range(len(L2)):
            P.append(np.kron(L1[i], L2[j]))
    return P

def KroneckerListOfLists(L):
    result = L[0]
    for x in L:
        result = KroneckerTwoSets(result, x)
    return result

def KroneckerMultiplyList(myList):
    result = [[1]]
    for x in myList:
        result = np.kron(result, x)
    return result

def angles(N):
    angles1 = np.zeros(N)
    angles1[0] = np.pi/4
    angles2 = np.zeros(N)
    angles2[0] = np.pi/4 + np.pi/2
    for i in range(1,N):
        angles2[i] = np.pi/2
    L = []    
    for i in range(N):
          L.append([angles1[i],angles2[i]])      
    return L, angles1, angles2

###############################################################################
### This function does the optimization of angles in Svetlichny inequality ####
###############################################################################

def SVL(rho,N,angles):
    OPS = []
    l = [0, 1]
    sigma_z = np.array([[1, 0], [0, -1]])
    M = list(itertools.product(l, repeat=N))
    for A in range(N):
        theta1 = angles[A][0]
        phi1 = angles[A][1]
        theta2 = angles[A][2]
        phi2 = angles[A][3]
        A1 =  theoretical_R(theta1,phi1) @ sigma_z @ theoretical_R(theta1,phi1).conj().T 
        A2 =  theoretical_R(theta2,phi2) @ sigma_z @ theoretical_R(theta2,phi2).conj().T 
        OPS.append([A1,A2])
    terms = []  
    for I in M:
        term = []
        for j in range(len(I)):
            a = OPS[j][I[j]]
            term.append(a)
        T = nu(sign(I))*KroneckerMultiplyList(term)
        terms.append(T)
    S = sum( t for t in terms )
    Svl= np.sqrt((np.trace(S@rho).astype(float))**2)
    return -Svl       

###############################################################################
##### This function implements the circuits needed for spin observables #######
###############################################################################

def SVL_circuits(circuit,N,angles):
    C_M = []
    l = [0, 1]
    M = list(itertools.product(l, repeat=N))
    for A in range(N):
        theta1 = angles[A][0]
        phi1 = angles[A][1]
        theta2 = angles[A][2]
        phi2 = angles[A][3]
        M1 = QuantumCircuit(N)          
        M1.r(-theta1,phi1, [A])

        M2 = QuantumCircuit(N)
        M2.r(-theta2,phi2, [A])
        C_M.append([M1,M2]) 
    sign_circuits = []  
    for I in M:
        correlation_circuit = QuantumCircuit(N)
        c = circuit.copy()
        list_c = []
        for j in range(len(I)):
            list_c.append(C_M[j][I[j]]) 
        result_circuit = correlation_circuit
        for element in list_c:
            result_circuit = result_circuit.compose(element)
            d = c.compose(result_circuit)        
        signo = nu(sign(I))
        sign_circuits.append([d,signo])
    return  sign_circuits       


####################################################
#### Mermin inequality functions ###################
####################################################


# Define iterative step in Mermin function

def Mermin_step(M):
    key_func = lambda x: x[1] 
    F0 = []
    F1 = []
    F2 = []
    F3 = []
    M_primed = []
    for x in M:
        t = []
        for j in range(len(x[1])):
            if x[1][j] == 0:
                t.append(1)
            else:
                t.append(0)    
        E = [x[0],t]
        M_primed.append(E)    
    for i in range(len(M)):
        T0 = [M[i][0],M[i][1]+[0]]
        T1 = [M[i][0],M[i][1]+[1]]
        T2 = [M_primed[i][0],M_primed[i][1]+[0]]
        T3 = [M_primed[i][0]*(-1),M_primed[i][1]+[1]] 
        F0.append(T0)
        F1.append(T1)
        F2.append(T2)
        F3.append(T3)
    M_step_0 = F0 + F1 + F2 + F3
    M_step = []
    for key, group in itertools.groupby(sorted(M_step_0, key=lambda x: x[1]), lambda x: x[1]):
        l = []
        for x in group:
            l.append(x[0])     
        s = sum(l)    
        if s != 0:
            T = [s, key] 
            M_step.append(T)
    return M_step

# Define the Mermin choice of observables for arbitrary N. 

def Mermin_choice(N):
   # Define first Mermin polynomial:
    M_0 = [[1,[0]]]
    L = M_0
    steps = N
    for _ in range(steps-1):
        L = Mermin_step(L)
    return L   

def Mermin(rho, N, angles):
    OPS = []
    sigma_z = np.array([[1, 0], [0, -1]])
    # get Mermin choice of observables with inequality coefficients 
    M = Mermin_choice(N)
    for A in range(N):
        theta1 = angles[A][0]
        phi1 = angles[A][1]
        theta2 = angles[A][2]
        phi2 = angles[A][3]
        A1 =  theoretical_R(theta1,phi1) @ sigma_z @ theoretical_R(theta1,phi1).conj().T 
        A2 =  theoretical_R(theta2,phi2) @ sigma_z @ theoretical_R(theta2,phi2).conj().T 
        OPS.append([A1,A2])
    terms = []  
    for I in M:
        term = []
        for j in range(len(I[1])):
            a = OPS[j][I[1][j]]
            term.append(a)
        T = I[0] * KroneckerMultiplyList(term)
        terms.append(T)
    M_operator = sum( t for t in terms )
    violation = np.sqrt((np.trace(M_operator@rho).astype(float))**2)
    violation *= 1/(2**(N-1))
    return -violation

def Mermin_circuits(circuit,N,angles):
    C_M = []
    M = Mermin_choice(N)
    for A in range(N):
        theta1 = angles[A][0]
        phi1 = angles[A][1]
        theta2 = angles[A][2]
        phi2 = angles[A][3]
        M1 = QuantumCircuit(N)
        M1.r(-theta1, phi1, [A])
        M2 = QuantumCircuit(N)
        M2.r(-theta2, phi2, [A])
        C_M.append([M1,M2]) 
    sign_circuits = []  
    for I in M: 
        correlation_circuit = QuantumCircuit(N)
        c = circuit.copy()
        list_c = []
        for j in range(len(I[1])):
            list_c.append(C_M[j][I[1][j]])     
        result_circuit = correlation_circuit
        for element in list_c:
            result_circuit = result_circuit.compose(element)
            d = c.compose(result_circuit)  
        coefficient = I[0]
        sign_circuits.append([d,coefficient])
    return  sign_circuits  


