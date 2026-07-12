from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np

from depolarizing_noise_model import *

def grover_oracle(marked_state):
    """Build a Grover oracle for multiple marked states. 
    We infer the qubit number from the length of the first marked state.

    Parameters:
        marked_state (str or list): Marked states of oracle

    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle
    """
    if not isinstance(marked_state, list):
        marked_state = [marked_state]
    # compute number of qubits from length of first marked state
    num_qubits = len(marked_state[0])

    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_state:
        # match qiskit endian encoding by reversing the target string
        rev_target = target[::-1]
        # identify indices of all the '0' elements in bit-string
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        # add multi controlled Z gate and X transforms
        qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        qc.x(zero_inds)
    return qc


def run_grover_and_compute_success_probability(qc, marked_state, shots, noise=None):
    """Run Grover circuit and compute success probability for marked states.
    Args:
        qc (QuantumCircuit): Grover's circuit
        marked_state (str or list): marked states for Grover's algorithm
        shots (int): number of shots for simulation
        noise (NoiseModel, optional): noise model for simulation. Defaults to None.
    Returns: 
        success probability of finding a marked state
    """
    if not isinstance(marked_state, list):
        marked_state = [marked_state]
    if noise is None:
        backend = AerSimulator()
        t_qc = transpile(qc, backend)
        result = backend.run(t_qc, shots=shots).result()
        counts = result.get_counts()

        # Reverse bitstrings due to Qiskit's little-endian ordering
        marked_set = {state for state in marked_state}
        success_counts = sum(counts.get(state, 0) for state in marked_set)
        success_prob = success_counts / shots
        print(f"Success probability of finding a marked state: {success_prob:.4f}")


    elif noise is not None:
        # noisy simulation 
        marked_set = {state for state in marked_state}
        backend = AerSimulator(noise_model=noise)
        t_qc = transpile(qc, backend)
        result = backend.run(t_qc, shots=shots).result()
        counts = result.get_counts()
        success_counts = sum(counts.get(state, 0) for state in marked_set)
        success_prob = success_counts / shots
        print(f"Success probability of finding a marked state with noise: {success_prob:.4f}")

    return success_counts, success_prob

def get_grover_circuit(marked_state):
    """get Grover's circuit with optimal number of iterations

    Args:
        marked_state (list of str): marked states for Grover's algorithm

    Returns:
        qc (QuantumCircuit): Grover's circuit with optimal number of iterations
        num_qubits (int): number of qubits in the circuit
    """
    if not isinstance(marked_state, list):
        marked_state = [marked_state]

    oracle = grover_oracle(marked_state)
    grover_op = grover_operator(oracle)
    qc = QuantumCircuit(grover_op.num_qubits)
    # create even superposition of all basis states
    qc.h(range(grover_op.num_qubits))
    # apply Grover operator the optimal number of times
    optimal_num_iterations = int(np.floor(np.pi / (4 * np.asin(np.sqrt(len(marked_state) / 2**grover_op.num_qubits)))))
    print("Optimal number of iterations: ", optimal_num_iterations)
    qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
    qc.measure_all()
    print("using ", optimal_num_iterations, " iterations for", grover_op.num_qubits, " qubits and ", len(marked_state), " marked states")

    oracle.draw(output = "mpl")
    grover_op.decompose().draw(output = "mpl")
    return qc, grover_op.num_qubits



if __name__ == "__main__":
    # number of shots 
    shots = 1e6

    # we mark one target state and build Grover's circuit
    marked_state = "11101"
    qc, num_qubits = get_grover_circuit(marked_state)

    return_data = []
    for noise in [None, noise_model_depolarizing(num_qubits=num_qubits, p_dep1=0.001)]:
        if noise is not None:
            noise_model, params_noise = noise
        elif noise is None:
            noise_model = None
            params_noise = None
        success_counts, success_prob = run_grover_and_compute_success_probability(qc, marked_state, shots=shots, noise=noise_model)
        return_data.append((success_counts, success_prob, noise))
