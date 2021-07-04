# adapt_ic-povm
Learning to measure: adaptive informationally complete POVMs for near-term quantum algorithms


## Installation
This repository requires Python 3 (tested with Python 3.8.5). Install the prerequisites with

```
pip install -r requirements.txt
```

## Usage

### Example Hamiltonians

The file `hamiltonians.pickle` contains a list of example chemistry Hamiltonians for the H2, LiH and H2O molecules,
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

The list of hamiltonians is shown in [Chemistry_hamiltonians_summary.ipynb]().

The file `hamiltonians.pickle` is generated with `python preprare_chemistry_hamiltonians.py`. It will take a considerable amount of time and a workstation or cluster is recommended.

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

