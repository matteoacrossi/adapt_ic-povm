# adapt_ic-povm
Learning to measure: adaptive informationally complete POVMs for near-term quantum algorithms


## Installation
This repository requires Python 3 (tested with Python 3.8.5). Install the prerequisites with

```
pip install -r requirements.txt
```

## Usage

The file `hamiltonians.pickle` contains a list of example chemistry Hamiltonians for the H2, LiH and H2O molecules,
with different bases, symmetries and fermion-to-qubit mappings. They are reported here:

|    | molecule   | basis   | two_qubit_reduction   | z2symmetry_reduction   | freeze_core   | mapping       |   qubits |   num_paulis |   vqe_value |   exact_value |
|---:|:-----------|:--------|:----------------------|:-----------------------|:--------------|:--------------|---------:|-------------:|------------:|--------------:|
|  0 | H2         | sto3g   | False                 | False                  | False         | parity        |        4 |           15 |   -1.84267  |      -1.84269 |
|  1 | H2         | sto3g   | False                 | False                  | False         | jordan_wigner |        4 |           15 |   -1.84269  |      -1.84269 |
|  2 | H2         | 631g    | True                  | False                  | False         | parity        |        6 |          159 |   -1.8516   |      -1.85726 |
|  3 | H2         | 631g    | False                 | False                  | False         | jordan_wigner |        8 |          185 |   -1.83211  |      -1.85726 |
|  4 | H2         | 631g    | False                 | False                  | False         | parity        |        8 |          185 |   -1.60457  |      -1.85726 |
|  5 | H2         | sto3g   | False                 | False                  | False         | bravyi_kitaev |        4 |           15 |   -1.84269  |      -1.84269 |
|  6 | H2         | 631g    | False                 | False                  | False         | bravyi_kitaev |        8 |          185 |   -1.60296  |      -1.85726 |
|  7 | LiH        | sto3g   | True                  | True                   | False         | parity        |        8 |          558 |   -8.87328  |      -8.87453 |
|  8 | LiH        | sto3g   | True                  | True                   | True          | parity        |        6 |          231 |   -0.950255 |      -1.07808 |
|  9 | LiH        | sto3g   | True                  | False                  | False         | parity        |       10 |          631 |   -8.8541   |      -8.87453 |
| 10 | LiH        | sto3g   | False                 | True                   | True          | bravyi_kitaev |        6 |          231 |   -1.05868  |      -1.07808 |
| 11 | LiH        | sto3g   | False                 | True                   | False         | bravyi_kitaev |        8 |          558 |   -8.85407  |      -8.87453 |
| 12 | H2O        | sto3g   | False                 | True                   | True          | parity        |        8 |          514 |  -22.4216   |     -23.5445  |
| 13 | H2O        | sto3g   | False                 | True                   | False         | parity        |       10 |         1035 |  -83.2595   |     -84.2064  |
| 14 | H2O        | sto3g   | False                 | True                   | False         | bravyi_kitaev |       10 |         1035 |  -82.7967   |     -84.2064  |
| 15 | H2O        | sto3g   | False                 | True                   | True          | bravyi_kitaev |        8 |          514 |  -23.5008   |     -23.5445  |
| 16 | H2O        | sto3g   | False                 | False                  | True          | jordan_wigner |       12 |          551 |  -23.4952   |     -23.5445  |
| 17 | LiH        | sto3g   | False                 | False                  | False         | parity        |       12 |          631 |   -8.8041   |      -8.87453 |
| 18 | H2         | sto3g   | False                 | False                  | False         | neven         |        4 |           15 |   -1.82173  |      -1.84269 |
| 19 | H2         | 631g    | False                 | False                  | False         | neven         |        8 |          185 |   -1.83818  |      -1.85726 |
| 20 | LiH        | sto3g   | False                 | False                  | False         | bravyi_kitaev |       12 |          631 |   -8.77588  |      -8.87453 |
| 21 | H2O        | sto3g   | False                 | False                  | True          | parity        |       12 |          551 |  -23.1052   |     -23.5445  |
| 22 | H2O        | sto3g   | False                 | False                  | True          | bravyi_kitaev |       12 |          551 |  -23.0872   |     -23.5445  |
| 23 | LiH        | sto3g   | False                 | False                  | False         | jordan_wigner |       12 |          631 |   -8.85405  |      -8.87453 |
| 24 | LiH        | sto3g   | False                 | False                  | False         | neven         |       12 |          631 |   -8.85407  |      -8.87453 |
| 25 | H2O        | sto3g   | False                 | False                  | True          | neven         |       12 |          551 |  -23.4985   |     -23.5445  |
| 26 | H2O        | sto3g   | False                 | False                  | False         | parity        |       14 |         1086 |  -84.1569   |     -84.2064  |
| 27 | H2O        | sto3g   | False                 | False                  | False         | bravyi_kitaev |       14 |         1086 |  -83.889    |     -84.2064  |
| 28 | H2O        | sto3g   | False                 | False                  | False         | jordan_wigner |       14 |         1086 |  -83.8552   |     -84.2064  |
| 29 | H2O        | sto3g   | False                 | False                  | False         | neven         |       14 |         1086 |  -83.5525   |     -84.2064  |
