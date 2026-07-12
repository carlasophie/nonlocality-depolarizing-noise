from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error
)

def noise_model_depolarizing(num_qubits=None, p_dep1=0.001):
    # fix noise parameters that are returned later for consistency    
    p_dep1 = p_dep1
    p_dep2 = 2*p_dep1 # Depolarizing error for two-qubit gates, set to twice the single-qubit depolarizing error
    
    # QuantumError objects
    error_depolarizing1 =  depolarizing_error(param = p_dep1, num_qubits = 1)
    error_depolarizing2 =  depolarizing_error(param = p_dep2, num_qubits = 2)

    # Add errors to noise model: all qubits
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(error_depolarizing1, ["u1", "u2", "u3"])
    noise.add_all_qubit_quantum_error(error_depolarizing2, ["cx"]) 
    # save parameters in a dict
    params = { "p_depolarizing1": p_dep1, "p_depolarizing2": p_dep2}

    print(noise)
    return noise, params 