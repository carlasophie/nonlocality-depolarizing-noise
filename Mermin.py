from tqdm import tqdm
import numpy as np 
from random import random
from scipy import optimize
from operator import itemgetter
import pandas as pd
from qiskit import transpile
import warnings
import matplotlib.pyplot as plt
import itertools
import pickle 
import os
from qiskit_aer import AerSimulator
from joblib import Parallel, delayed

#costum import 
from circuit_functions import *



def task(RHO, N, seeds = 40):
    """
    Compute maximum Mermin violation angles for a given element RHO as input.

    Args:
        RHO (tuple): Single element consisting of its density matrix and circuit.
        N (int): Number of qubits.
        seeds (int, optional): Number of iterations for optimization. Defaults to 40.

    Returns:
        value (tuple): Maximum violation value, fitted parameters, and circuit.
        seeds (int): Number of seeds used in optimization.
    """
    seeds = seeds 
    warnings.filterwarnings('ignore')
    def f(params):
        angs = [params[4 * i:4 * (i + 1)] for i in range(N)]
        return Mermin(RHO[0],N,angs)           
    partial_results = []
    
    done_seeds = 0
    while done_seeds < seeds:
        initial_guess = [] 
        for _ in range(4*N):
            y =  2*(np.pi)*(random())
            initial_guess.append(y)
        bnds = list(itertools.repeat((0, 2*np.pi),4*N))    
        res = optimize.minimize(f, initial_guess, bounds = bnds)

        # store results if optimization was successful
        if res.success:
            fitted_params = res.x
            if np.all(fitted_params):
                done_seeds += 1
                partial_results.append([-f(res.x),fitted_params,RHO[1]])  

    value = max(partial_results, key=itemgetter(0))
    return value, seeds

def get_circuits(dataset_name, param_set_name, seeds=40, print_example=False):
    """"
    Generate circuits at directory 'circuits_for_experiments_Mermin/', 
    apply the task function the every element in list states_0.
    
    Args:
        dataset_name (str): path to the pickle file containing the dataset of states.
        param_set_name (str): path to the pickle file containing the circuit parameters.
        seeds (int, optional): number of optimization iterations. Defaults to 40.
        print_example (bool, optional): whether to print an example of the saved experiment_dict_j.pkl files. Defaults to False.
    Returns:
        circuits_for_backends (list): list of dictionaries containing circuits and theoretical violation values.
        circ_params (dict): dictionary of circuit parameters.
        states_0 (list): list of initial states.
        seeds (int): number of optimization iterations used.
    """

    # get dataset of states and parameters from pickle file
    with open(dataset_name, 'rb') as f:
        states_0 = pickle.load(f)
    # load parameters from pickle file
    with open(param_set_name, 'rb') as f:
        circ_params = pickle.load(f)
    
    #extract the stamp (no. between stamp_ and .pkl) from dataset_name
    stamp = dataset_name.split('stamp_')[1].split('.pkl')[0]
    print("stamp:", stamp)

    
    N = circ_params['N']
    n_circuits = len(states_0)

    # Lists where we store the circuits, parmeters obtained, and the violation values.
    circuits = []
    parameters = []
    values = []
    # run parallelized task
    results = Parallel(n_jobs=-1)(delayed(task)(x, N, seeds) for x in tqdm(states_0))
    results, seeds = zip(*results)  # Unpack results and seeds
    # unpack the results
    values, parameters, circuits = zip(*results)

    # extract folder name
    folder_path = os.path.dirname(dataset_name)
    folder_name = os.path.basename(folder_path)

    circuits_for_backends =  []

    if not os.path.exists('circuits_for_experiments_Mermin'):
        os.makedirs('circuits_for_experiments_Mermin')
    os.makedirs("circuits_for_experiments_Mermin/"+folder_name, exist_ok=True)

    for j in tqdm(range(len(circuits))):
        experiment_dict = {}
        angles = [parameters[j][4 * i:4 * (i + 1)] for i in range(N)] 
        c = circuits[j]
        value = values[j]
        circuits_m = Mermin_circuits(c,N,angles)
        # theoretical violation value
        experiment_dict["Theoretical_violation"] = value
        # circuits with the sign as their correlations have in the Mermin inequality.
        experiment_dict["sign_circuits"] = circuits_m
        circuits_for_backends.append(experiment_dict)
        with open('circuits_for_experiments_Mermin/' + folder_name + '/experiment_dict_' + str(j) + '.pkl', 'wb') as f:
            pickle.dump(experiment_dict, f)
            
    # example structure of one of the experiment_dict_j.pkl files
    if print_example:
        with open('circuits_for_experiments_Mermin/' + folder_name + '/experiment_dict_0.pkl', 'rb') as f:
            experiment_data = pickle.load(f)    

        print("experimental data:", experiment_data)    
    return circuits_for_backends, circ_params, states_0, seeds

def run_ideal_simulator_mermin(dataset_name, param_set_name, seeds=40, circuits_for_backends=None, circ_params=None, states_0=None):
    """
    Run ideal simulator for Mermin inequality circuits.
    Args:
        dataset_name (str): path to the pickle file containing the dataset of states.
        param_set_name (str): path to the pickle file containing the circuit parameters.
        seeds (int, optional): number of optimization iterations. Defaults to 40.
        circuits_for_backends (list, optional): list of dictionaries containing circuits and theoretical violation values. Defaults to None.
        circ_params (dict, optional): dictionary of circuit parameters. Defaults to None.
        states_0 (list, optional): list of initial states. Defaults to None.
    Returns:
        df (DataFrame): DataFrame containing the results of the ideal simulation.
        filename (str): path to the CSV file where results are saved.
    """
    #extract the stamp (no. between stamp_ and .pkl) from dataset_name
    stamp = dataset_name.split('stamp_')[1].split('.pkl')[0]
    print("stamp:", stamp)

    if circuits_for_backends is None or circ_params is None or states_0 is None:
        # avoid recomputation if the circuits are already provided
        circuits_for_backends, circ_params, states_0, seeds = get_circuits(dataset_name, param_set_name, seeds=seeds)

    
    N = circ_params['N']
    n_circuits = len(states_0)
    print("Number of circuits:", n_circuits)

    simulator = AerSimulator()
    # number of shots
    S = 10000
    os.makedirs("../results_nonloc/plots", exist_ok=True)

    results_ideal = []
    theo_value = []
    for x in circuits_for_backends:
            correlations = []
            for i in range(len(x['sign_circuits'])):
                p_plus = 0
                p_minus = 0
                circuit = x['sign_circuits'][i][0].copy()
                circuit.measure_all()
                circ = transpile(circuit, simulator)
                result = simulator.run(circ, shots = S).result()
                counts = result.get_counts(circ)
                for key in counts:
                    count = key.count("1")
                    if (-1)**(count) < 0:
                        p_minus += counts[key]/S
                    else:
                        p_plus += counts[key]/S  
                pre_term =  p_plus - p_minus
                term = (x['sign_circuits'][i][1])*pre_term
                correlations.append(term)
                del(circuit)
                del(circ)
            value = x["Theoretical_violation"]
            mean = np.abs(sum(correlations))
            mean *= 1/(2**(N-1))  # Scale the mean by 2^(N-1) for Mermin inequality
            results_ideal.append(mean)
            theo_value.append(value)
            print("mean:", mean, "theoretical value:", value)

    #create folder
    os.makedirs("../results_nonloc", exist_ok=True)
    os.makedirs("../results_nonloc/results_mermin", exist_ok=True)
    
    #save resuls_ideal to df
    df = pd.DataFrame(results_ideal, columns=["Mermin_violation"])
    df["theoretical_value"] = theo_value
    df["dataset_name"] = [str(dataset_name)]* len(results_ideal)
    df["D"] = [circ_params['D']]*len(results_ideal)
    df["seed_rounds"] = seeds
    filename = "../results_nonloc/results_mermin/results_ideal_mermin_N_" + str(N) + "_n_circuits_{}_stamp_{}.csv".format(n_circuits, stamp)
    df.to_csv(filename, index=False)

    plt.figure()
    plt.title("Mermin inequality violations for ideal circuits")
    plt.hist(results_ideal,bins=30)
    plt.xlabel("Mermin violation value")
    plt.ylabel("Frequency")
    plt.title("Mermin inequality violations for ideal circuits with {} qubits".format(circ_params['N']))
    plt.savefig('../results_nonloc/plots/histo_ideal_mermin_N_' + str(N) + '_n_circ_' + str(n_circuits) + "N_qubits" + str(circ_params['N']) + 'stamp' + str(stamp) + '.png')

    return df, filename


if __name__ == "__main__":
    
    # define test dataset: example paths
    dataset_name = 'data_states/QRC_Marmot_N_3_n_circuits_500/states_0_stamp_887602.pkl'
    param_set_name = 'data_states/QRC_Marmot_N_3_n_circuits_500/params_stamp_887602.pkl'

    circuits_for_backends, circ_params, states_0, seeds = get_circuits(dataset_name, param_set_name)
    df = run_ideal_simulator_mermin(dataset_name, param_set_name, seeds=40, circuits_for_backends=circuits_for_backends, circ_params=circ_params, states_0=states_0)