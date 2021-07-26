from numpy.random import seed
from qiskit.chemistry.transformations import (
    FermionicTransformation,
    FermionicTransformationType,
    FermionicQubitMappingType,
)
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


def h_chain(n_qubits, hh_distance=0.75, basis="sto3g", he_mode=False):
    """Generates a Hydrogen chain to be simulated with a specific number of qubits.

    Args:
        n_qubits (int): number of qubits
        hh_distance (float, optional): distance between Hydrogen atoms. Defaults to 0.75.
        basis (str, optional): Molecular basis, can be 'sto3g' or '631g'. Defaults to 'sto3g'.
        he_mode (bool, optional): replace hydrogen with helium

    Raises:
        Exception: if basis is not 'sto3g' or '631g', or if n_qubits is not divisible by 2 (or 4 in case of 631g)

    Returns:
        QMolecule: a QMolecule object from which you can get .one_body_integrals and .two_body_integrals
    """

    assert n_qubits >= 4, "n_qubits must be an integer >= 4 and divisible by 2"
    assert type(n_qubits == int), "n_qubits must be an integer >= 4 and divisible by 2"
    assert n_qubits % 2 == 0, "n_qubits must be an integer >= 4 and divisible by 2"

    # get number of H atoms
    if basis == "sto3g":
        num_hydrogens = n_qubits // 2
    elif basis == "631g":
        assert (
            n_qubits >= 8
        ), "n_qubits must be an integer >= 8 and divisible by 4 in the 631g basis"
        assert (
            n_qubits % 4 == 0
        ), "n_qubits must be an integer >= 8 and divisible by 4 in the 631g basis"
        num_hydrogens = n_qubits // 4
    else:
        raise Exception("basis must be either 'sto3g' or '631g' ")

    if he_mode == True:
        atom = "He"
    else:
        atom = "H"
    # generate geometry of a hydrogen chain
    geom = [[atom, [0.0, 0.0, k * hh_distance]] for k in range(num_hydrogens)]
    # generate a molecule object and run it through the driver
    molecule = Molecule(geometry=geom, charge=0, multiplicity=len(geom) % 2 - 1)
    return molecule


def prepare_hamiltonian(qubits, mapping):
    molecule = h_chain(qubits)

    driver = PySCFDriver(molecule=molecule, unit=UnitsType.ANGSTROM, basis="sto3g")

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
        two_qubit_reduction=False,
        z2symmetry_reduction=None,
        freeze_core=False,
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
    print("Constructing exact state")
    exact_eigensolver = NumPyEigensolver(qubitOp)
    print(qubitOp.to_dict())
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
        str(result_final["qubits"]) + "q " + "h_chain" + " " + result_final["mapping"]
    )

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
    hamiltonian_parameters = []
    for qubits in range(4, 16, 2):
        for mapping in ["parity", "bravyi_kitaev", "neven", "jordan_wigner"]:
            hamiltonian_parameters.append({"qubits": qubits, "mapping": mapping})

    filename = "data/hamiltonians_h_chain.pickle"

    if os.path.exists(filename):
        print(f"File {filename} exists.")
        exit()

    results = Parallel(n_jobs=8)(
        delayed(process_params)(h, filename) for h in hamiltonian_parameters
    )

    print("All done. Bye.")
