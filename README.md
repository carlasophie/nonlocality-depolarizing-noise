# Investigating nonlocality under depolarizing noise

[![arXiv](https://img.shields.io/badge/arXiv-2610.24196-b31b1b.svg)](https://arxiv.org/abs/2610.24196)

If you use this code, please cite the corresponding paper:

| Corresponding Paper | 
|----------|
| Granda Arango, A., Rieger, C., et al. "Nonlocality as a Benchmarking Tool for Quantum Processors", accepted for publication in EPJ Quantum Technology. | 

*Code contributors: Carla Rieger, Federico Hernán Holik, Camilo Andrés Granda Arango.*

---

## Installation
```
python3.11 -m venv nonloc_dp
source nonloc_dp/bin/activate
pip install -r requirements.txt
```
---

## Usage 
### Get Svetlichny inequality results (ideal and noisy)
```
python3 Svetlichny.py
python3 Svetlichny_noise.py
```
*this will produce .csv files in the '../results_nonloc/results_svet/' folder and histogram figures in '../results_nonloc/plots/', the .csv file contains also the mean violation (simulator and theoretical);*
*make sure to use the right path to the correct dataset and parameter set by changing 'dataset_name' generated from random circuit instances created by QRC.py and 'param_set_name' with the corresponding parameter set file*

### Get Mermin inequality results (ideal and noisy)
```
python3 Mermin.py
python3 Mermin_noise.py
```
*this will produce .csv files in the '../results_nonloc/results_mermin/' folder and histogram figures in '../results_nonloc/plots/', the .csv file contains also the mean violation (simulator and theoretical);*
*make sure to use the right path to the correct dataset and parameter set by changing 'dataset_name' generated from random circuit instances created by QRC.py and 'param_set_name' with the corresponding parameter set file*

### Get performance metrics for Grover on noise models from before (ideal and noisy) for a specified number of qubits
```
python3 Grover.py
```
