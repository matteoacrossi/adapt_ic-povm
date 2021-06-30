from numpy.random import seed
from qiskit.chemistry.transformations import (
    FermionicTransformation,
    FermionicTransformationType,
    FermionicQubitMappingType,
)
from qiskit.chemistry.drivers import Molecule
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule

from qiskit import Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE, NumPyEigensolver

from tqdm.notebook import tqdm

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import json
import pickle
import os
from filelock import FileLock

molecules = {
    "H2": Molecule(
        geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.75]]],
        charge=0,
        multiplicity=1,
    ),
    "LiH": Molecule(
        geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.6]]],
        charge=0,
        multiplicity=1,
    ),
    "H2O": Molecule(
        geometry=[
            ["O", [0.0, 0.0, 0.0]],
            ["H", [0.757, 0.586, 0.0]],
            ["H", [-0.757, 0.586, 0.0]],
        ],
        charge=0,
        multiplicity=1,
    ),
}


def prepare_hamiltonian(
    molecule, basis, two_qubit_reduction, z2symmetry_reduction, freeze_core, mapping
):
    molecule = molecules[molecule]

    driver = PySCFDriver(molecule=molecule, unit=UnitsType.ANGSTROM, basis=basis)

    if mapping == "parity":
        qubit_mapping = FermionicQubitMappingType.PARITY
    elif mapping == "bravyi_kitaev":
        qubit_mapping = FermionicQubitMappingType.BRAVYI_KITAEV
    elif mapping == "neven":
        qubit_mapping = FermionicQubitMappingType.NEVEN
    elif mapping == "jordan_wigner":
        qubit_mapping = FermionicQubitMappingType.JORDAN_WIGNER
    else:
        raise ValueError("Wrong mapping")

    fermionic_transformation = FermionicTransformation(
        transformation=FermionicTransformationType.FULL,
        qubit_mapping=qubit_mapping,
        two_qubit_reduction=two_qubit_reduction,
        z2symmetry_reduction="auto" if z2symmetry_reduction else None,
        freeze_core=freeze_core,
    )

    qubitOp, _ = fermionic_transformation.transform(driver)

    return qubitOp


def construct_operator(params):
    """Construct the legacy operator and collect info"""
    op = prepare_hamiltonian(**params)

    result = {
        **params,
        **{
            "operator": op.to_legacy_op(),
            "qubits": op.num_qubits,
            "num_paulis": len(op),
        },
    }
    return result


def run_vqe(qubitOp):
    aqua_globals.random_seed = 0
    """Find the optimal ground state using predefined VQE ansatz"""
    quantum_instance = QuantumInstance(
        backend=Aer.get_backend("statevector_simulator"),
        seed_simulator=0,
        seed_transpiler=0,
    )
    algo = VQE(qubitOp)
    vqe_result = algo.run(quantum_instance)
    qc = algo.get_optimal_circuit()
    exact = np.real(vqe_result["eigenvalue"])
    return {"vqe_circuit": qc, "vqe_value": exact}


def run_exact(qubitOp):
    """Find the ground state using exact diagonalization"""
    exact_eigensolver = NumPyEigensolver(qubitOp)
    res = exact_eigensolver.run()
    # E_ex.append(np.real(res['eigenvalues'][0]))
    qc = res["eigenstates"][0].to_circuit_op().to_circuit()
    exact = np.real(res["eigenvalues"][0])
    return {"exact_circuit": qc, "exact_value": exact}


def add_vqe_exact(params):
    """Add the ground states to the dict"""
    ex = run_exact(params["operator"])
    vq = run_vqe(params["operator"])

    return {**params, **vq, **ex}


def process_params(params, filename):
    print("Constructing operator for", *(params.values()))
    result = construct_operator(params)
    print("Constructing states for", *(params.values()))
    result_final = add_vqe_exact(result)
    print("Done", *(params.values()))
    # return result_final

    result_final["name"] = (
        str(result_final["qubits"])
        + "q "
        + params["molecule"]
        + " "
        + result_final["mapping"]
    )

    # df.append(result_final)
    with FileLock(f"{filename}.lock"):
        try:
            with open(filename, "rb") as file:
                hams = pickle.load(file)
        except FileNotFoundError:
            hams = []

        hams.append(result_final)
        with open(filename, "wb") as file:
            pickle.dump(hams, file)
        print(f"Written to {filename}")


if __name__ == "__main__":
    hamiltonian_parameters = json.load(open("hamiltonians_params.json"))

    filename = "hamiltonians.pickle"

    if os.path.exists(filename):
        print(f"File {filename} exists.")
        exit()

    results = Parallel(n_jobs=8)(
        delayed(process_params)(h, filename) for h in hamiltonian_parameters
    )

    print("All done. Bye.")
