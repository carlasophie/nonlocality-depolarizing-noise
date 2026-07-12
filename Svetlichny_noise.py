import numpy as np 
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import os
from qiskit_aer import AerSimulator
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from circuit_functions import *
from depolarizing_noise_model import noise_model_depolarizing
from calibration_func import *


def get_data_svet(dataset_name, param_set_name):
    """Get data for Svetlichny inequality experiments.

    Args:
        dataset_name (str): path to the pickle file containing the dataset of states.
        param_set_name (str): path to the pickle file containing the circuit parameters.

    Returns:
        N (int): number of qubits.
        dataset_name (str): path to the dataset file.
        param_set_name (str): path to the parameter set file.
        states_0 (list): list of initial states.
        circ_params (dict): dictionary of circuit parameters.
        n_circuits (int): number of circuits.
        circuits_for_backends (list): list of dictionaries containing circuits and theoretical violation values.
        stamp (str): unique stamp extracted from dataset_name.
    """
    #extract the stamp (no. between stamp_ and .pkl) from dataset_name
    stamp = dataset_name.split('stamp_')[1].split('.pkl')[0]
    print("stamp:", stamp)

    folder_path = os.path.dirname(dataset_name)
    folder_name = os.path.basename(folder_path)


    # load states from pickle file
    with open(dataset_name, 'rb') as f:
        states_0 = pickle.load(f)
    # load parameters from pickle file
    with open(param_set_name, 'rb') as f:
        circ_params = pickle.load(f)

    N = circ_params['N']
    n_circuits = len(states_0)
    print("Number of circuits:", n_circuits)

    # load circuits from pickle file
    circuits_for_backends = []
    for j in range(n_circuits):
        with open('circuits_for_experiments_svet/' + folder_name + '/experiment_dict_' + str(j) + '.pkl', 'rb') as f:
            single_dict = pickle.load(f)
        circuits_for_backends.append(single_dict)
    return N, dataset_name, param_set_name, states_0, circ_params, n_circuits, circuits_for_backends, stamp




def run_noisy_simulator_svet(noise_model, params, dataset_name, param_set_name, states_0, circ_params, n_circuits, circuits_for_backends, stamp):
    """
    Run noisy simulation for Svetlichny inequality circuits.
    Args:
        noise_model (NoiseModel): noise model to be used in the simulation.
        params (dict): dictionary of parameters for the simulation.
        dataset_name (str): path to the dataset file.
        param_set_name (str): path to the parameter set file.
        states_0 (list): list of initial states.
        circ_params (dict): dictionary of circuit parameters.
        n_circuits (int): number of circuits.
        circuits_for_backends (list): list of dictionaries containing circuits and theoretical violation values.
        stamp (str): unique stamp extracted from dataset_name.
    """
    # create noisy simulator backend
    noise = noise_model
    sim_noise = AerSimulator(noise_model=noise)
    # number of shots
    S = 1000

    results_noise = []
    theo_value = []
    N = circ_params['N']

    for x in circuits_for_backends:
            correlations = []
            for i in range(len(x['sign_circuits'])):
                p_plus = 0
                p_minus = 0
                circuit_0 = x['sign_circuits'][i][0].copy()
                circuit_0.measure_all()
                # transpile and simulate with given noise model and AerSimulator
                circ_0 = transpile(circuit_0, sim_noise)
                result = sim_noise.run(circ_0, shots = S).result()
                counts = result.get_counts(circ_0)
                for key in counts:
                    count = key.count("1")
                    if (-1)**(count) < 0:
                        p_minus += counts[key]/S
                    else:
                        p_plus += counts[key]/S  
                pre_term =  p_plus - p_minus
                term = (x['sign_circuits'][i][1])*pre_term
                correlations.append(term)
                del(circ_0)
                del(circuit_0)
            value = x["Theoretical_violation"]
            theo_value.append(value)
            mean = np.abs(sum(correlations))
            results_noise.append(mean)
            print(mean, value)

    
    #create folder
    os.makedirs("../results_nonloc", exist_ok=True)
    os.makedirs("../results_nonloc/plots", exist_ok=True)
    name = "_p_depol1=" + str(params["p_depolarizing1"]) + "_p_depol2=" + str(params["p_depolarizing2"])
    plt.figure()
    plt.hist(results_noise,bins=30)
    plt.xlabel("Svetlichny violation value")
    plt.ylabel("Frequency")
    plt.title("Svetlichny inequality violations for noisy circuits with {} qubits".format(circ_params['N']) ) 
    plt.savefig("../results_nonloc/plots/histo_noise_svet_N_" + str(N) + "_n_circ_" + str(n_circuits) + 'stamp' + str(stamp) + '_noise_model_' + name + '.png')

    
    os.makedirs("../results_nonloc/results_svet", exist_ok=True)
    #save resuls_ideal to df together with parameters
    df = pd.DataFrame(results_noise, columns=["Svetlichny_violation"])
    df["theoretical_value"] = theo_value
    df["dataset_name"] = [str(dataset_name)]* len(results_noise)
    df["p_depolarizing1"] = [str(params["p_depolarizing1"])]*len(results_noise)
    df["p_depolarizing2"] = [str(params["p_depolarizing2"])]*len(results_noise)
    df["D"] = [circ_params['D']]*len(results_noise)

    #save df to csv
    filepath = "../results_nonloc/results_svet/results_noise_svetlichny_N_" + str(N) + "_n_circuits_{}_stamp_{}".format(n_circuits, stamp) + "_p_depol1=" + str(params["p_depolarizing1"]) + "_p_depol2=" + str(params["p_depolarizing2"]) + ".csv"

    df.to_csv(filepath, index=False) 
    return df, filepath


if __name__ == "__main__":

    # define test dataset: example paths
    dataset_name = 'data_states/QRC_Marmot_N_3_n_circuits_500/states_0_stamp_887602.pkl'
    param_set_name = 'data_states/QRC_Marmot_N_3_n_circuits_500/params_stamp_887602.pkl'

    # evaluate noisy simulations
    N, dataset_name, param_set_name, states_0, circ_params, n_circuits, circuits_for_backends, stamp = get_data_svet(dataset_name, param_set_name)
    for noisy_model in [noise_model_depolarizing(num_qubits=N, p_dep1=0.01)]:
        start_time = time.time()
        noise_model, params_noise = noisy_model
        run_noisy_simulator_svet(noise_model, params_noise, dataset_name, param_set_name, states_0, circ_params, n_circuits, circuits_for_backends, stamp)
        print("Noise model parameters:", noisy_model)
        end_time = time.time()
        print("Execution time:", end_time - start_time, "seconds")