# adapt_ic-povm
Code used to produce the results of the paper
G. García-Pérez, M. A. C. Rossi, B. Sokolov, F. Tacchino, P. K. Barkoutsos, G. Mazzola, I. Tavernelli, S. Maniscalco, "Learning to Measure: Adaptive Informationally Complete Generalized Measurements for Quantum Algorithms", [PRX Quantum 2, 040342 (2021)](https://doi.org/10.1103/PRXQuantum.2.040342).

The code uses qiskit 0.23 for simulating a near term quantum device, generating qubit Hamiltonian for small molecules and running VQE.

> NOTE: The simulations are quite demanding, especially for large molecules. The data was generated using a HPC cluster, and is openly available on [Zenodo.org](https://doi.org/10.5281/zenodo.5759030).

## Installation
This repository requires Python 3 (tested with Python 3.8.5). Since we need to patch the qiskit aqua module in order to add the JKMN mapping, we recommend creating a virtual environment.

Clone the repository and move to the folder

```
git clone https://github.com/matteoacrossi/adapt_ic-povm.git
cd adapt_ic-povm
```

Install the prerequisites with

```
pip install -r requirements.txt
```

Apply the patch to add the JKMN mapping

```
pypatch apply neven.patch qiskit
```

## Usage

### Downloading the data

The data used to reproduce the results presented in the paper is available at [https://doi.org/10.5281/zenodo.5759030](https://doi.org/10.5281/zenodo.5759030). Download the
file `data.tar` and extract it in the repository folder:

```
wget https://zenodo.org/record/5759030/files/data.tar
tar -xvf data.tar
```

A folder `data/` will be created.

### Example Hamiltonians

The example Hamiltonians are generated with `python preprare_chemistry_hamiltonians.py`. It will take a considerable amount of time and a workstation or cluster is recommended.
The molecules are defined in the file `preprare_chemistry_hamiltonians.py`.

The file `data/hamiltonians.pickle` contains a list of example chemistry Hamiltonians for the H2, LiH and H2O molecules,
with different bases, symmetries and fermion-to-qubit mappings. For each Hamiltonian, the exact ground state and an approximated
ground state found with VQE (using qiskit 0.23.6 default settings) are stored in the file.

Each entry in the file is a dictionary of the form:

```python
{'molecule': 'H2',
 'basis': 'sto3g',
 'two_qubit_reduction': False,
 'z2symmetry_reduction': False,
 'freeze_core': False,
 'mapping': 'parity',
 'operator': <qiskit.aqua.operators.legacy.weighted_pauli_operator.WeightedPauliOperator at 0x2b3a9078b9a0>,
 'qubits': 4,
 'num_paulis': 15,
 'vqe_circuit': <qiskit.circuit.library.n_local.real_amplitudes.RealAmplitudes at 0x2b3a8febd970>,
 'vqe_value': -1.842665192696192,
 'exact_circuit': <qiskit.circuit.quantumcircuit.QuantumCircuit at 0x2b3a8f0747f0>,
 'exact_value': -1.842686681905733,
 'name': '4q H2 parity'}
 ```

Similarly, the file `data/hamiltonians_h_chain.pickle` contains a list of Hamiltonians for hydrogen chains of various length. The file can be generated with `python prepare_h_chain_hamiltonians.py`.


The list of hamiltonians is shown in [Hamiltonians_summary.ipynb](./Hamiltonians_summary.ipynb).



### Simulations

The simulations can be executed with `run_simulation.py`. The file takes a number of arguments as input

```
usage: run_simulation.py [-h] [--file FILE] [--hamiltonian HAMILTONIAN] [--samples SAMPLES] [--shots SHOTS [SHOTS ...]] [--method METHOD] [--outfile OUTFILE] [--exact] [--counts]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE                   The file with the Hamiltonians (hamiltonians.pickle)
  --hamiltonian HAMILTONIAN     The id of the Hamiltonian
  --samples SAMPLES             How many samples to take
  --shots SHOTS [SHOTS ...]     The shots to use for the measurement (multiple amounts of shots can be passed)
  --method METHOD               One of the methods: SIC-POVM, Pauli, Grouped_Pauli, Grad-POVM, Google-POVM, Grad-Google-POVM
  --outfile OUTFILE             File where to save the results
  --exact                       Use the exact diagonalization state
  --counts                      Store the measurement counts to file (for tomography)
  ```

The output files are in the JSON lines format:

```json
{"qubits": 4, "true": -1.8426866160309918, "estimate": -1.8394723164727411, "estimated_error": 0.01613556904575243, "error": 0.0032142995582506995, "circuits": 15, "shots_per_circuit": 66, "shots": 990, "time_qc": 2.879150152206421, "time_post": 0.5117971897125244, "method": "Pauli", "counts": null, "name": "4q H2 jordan_wigner", "commit": "d59fe99", "exact_ground_state": false, "id": "MzZya4nhC6fQdvZsob332H", "timestamp": 1625040177.894622}
{"qubits": 4, "true": -1.8426866160309918, "estimate": -1.8418662180482184, "estimated_error": 0.01523071106596722, "error": 0.0008203979827734464, "circuits": 15, "shots_per_circuit": 66, "shots": 990, "time_qc": 2.938889980316162, "time_post": 0.522057294845581, "method": "Pauli", "counts": null, "name": "4q H2 jordan_wigner", "commit": "d59fe99", "exact_ground_state": false, "id": "Xd9n4GGivADMxi8fT6CuZJ", "timestamp": 1625040177.956557}
...
```

and can be easily processed with Pandas.

A convenience script `process_raw_data.py` joins multiple files, adds Hamiltonian information and saves the final dataset to a binary `.feather` file. For example:

```
python process_raw_data.py 'raw_data/*.txt' --hamiltonians hamiltonians.pickle -o data/chemistry_data.feather
```


### Tomography

For producing the results related to k-RDM tomography, all the counts coming from the measurements need to be stored. This can be achieved with the `--counts` flag of `run_simulation.py`.

The data used for the paper is stored in the JSON Lines file `counts_data.txt`. The data can be processed using `python tomography_script.py`.

### Figures

The notebook [Figures.ipynb](./Figures.ipynb) generates all the figures contained in the published paper from the data available at [https://doi.org/10.5281/zenodo.5759030](https://doi.org/10.5281/zenodo.5759030).

## Citation

G. García-Pérez, M. A. C. Rossi, B. Sokolov, F. Tacchino, P. K. Barkoutsos, G. Mazzola, I. Tavernelli, S. Maniscalco, "Learning to Measure: Adaptive Informationally Complete Generalized Measurements for Quantum Algorithms", [*PRX Quantum* **2**, 040342 (2021)](https://doi.org/10.1103/PRXQuantum.2.040342).

```
@misc{garciaperez2021learning,
        title = {Learning to Measure: Adaptive Informationally Complete Generalized Measurements for Quantum Algorithms},
  author = {Garc\'{\i}a-P\'erez, Guillermo and Rossi, Matteo A.C. and Sokolov, Boris and Tacchino, Francesco and Barkoutsos, Panagiotis Kl. and Mazzola, Guglielmo and Tavernelli, Ivano and Maniscalco, Sabrina},
  journal = {PRX Quantum},
  volume = {2},
  issue = {4},
  pages = {040342},
  numpages = {17},
  year = {2021},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/PRXQuantum.2.040342},
  url = {https://link.aps.org/doi/10.1103/PRXQuantum.2.040342}
}
```
