""" POVM operator """

import logging
from typing import List, Optional, Tuple, Union
from copy import deepcopy

from qiskit.quantum_info import Pauli
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.aqua import AquaError

import numpy as np
from joblib import Parallel, delayed
from joblib import parallel_backend

from time import time

logger = logging.getLogger(__name__)

# The parameters for the SIC POVM
_SIC_PARAMS = np.array([0.25, 0.30408672, 0.125, 0.5, 0.5, 0.25, 0.5, 0.75])

# Pauli matrices
_PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


class POVMOperator(WeightedPauliOperator):
    """A weighted POVM based operator built on the WeightedPauliOperator class"""

    def __init__(
        self,
        paulis: List[List[Union[complex, Pauli]]],
        basis: Optional[List[Tuple[object, List[int]]]] = None,
        z2_symmetries: "Z2Symmetries" = None,
        atol: float = 1e-12,
        name: Optional[str] = None,
        povm_params=None,
    ) -> None:
        """
        Args:
            paulis: the list of weighted Paulis or a WeightedPauliOperator, where a weighted pauli is
                    composed of a length-2 list and the first item is the
                    weight and the second item is the Pauli object.
            basis: the grouping basis, each element is a tuple composed of the basis
                    and the indices to paulis which belong to that group.
                    e.g., if tpb basis is used, the object will be a pauli.
                    By default, the group is equal to non-grouping, each pauli is its own basis.
            z2_symmetries: recording the z2 symmetries info
            atol: the threshold used in truncating paulis
            name: the name of operator.
            povm_params: the 3 + 3 + 2 = 8 parameters describing a POVM through
                         the initial preparation of the probe qubit,
                         the Cartan representation of the unitary coupling,
                         and the projective measurement at the output, respectively
                         (from arXiv:1407.0056), bounded by [0,1]
        """

        if isinstance(paulis, WeightedPauliOperator):
            self._basis = basis
            self._z2_symmetries = z2_symmetries
            self._name = name if name is not None else ""

            # plain store the paulis, the group information is store in the basis
            self._paulis_table = paulis._paulis_table.copy()
            self._paulis = paulis._paulis.copy()
            self._basis = paulis._basis.copy()
            self._atol = paulis._atol
        else:
            # TODO: This becomes very slow when there are many Pauli strings, due
            # to a simplification happening in the basis class init.
            super().__init__(paulis, basis, z2_symmetries, atol, name)

        self._num_qubits = super().num_qubits

        paulis_dict = super().to_dict()
        for o in paulis_dict["paulis"]:
            o["coeff"] = o["coeff"]["real"] + 1j * o["coeff"]["imag"]

        self._k = [o["label"] for o in paulis_dict["paulis"]]
        self._c_k = np.array([o["coeff"] for o in paulis_dict["paulis"]])

        if povm_params is None:
            povm_params = self._num_qubits * list(_SIC_PARAMS)

        if isinstance(povm_params, (list, np.ndarray)):
            if len(povm_params) != 8 * self._num_qubits:
                raise ValueError(
                    "The size of povm_params don't match the size of paulis"
                )
            self._povm_params = {
                i: np.asarray(povm_params[8 * i : 8 * (i + 1)])
                for i in range(self._num_qubits)
            }
        elif isinstance(povm_params, dict):
            self._povm_params = povm_params
        else:
            raise NotImplementedError

        self._povms = None
        self._b_matrices = None
        self._unitaries = None

        try:
            self._set_povm()
        except Exception as ex:
            print("POVM parameter array:", self.param_array)
            raise (ex)

    @property
    def num_qubits(self):
        """Number of qubits of the operator (excluding auxiliary qubits)

        Returns:
            int: number of qubits
        """
        return self._num_qubits

    @property
    def povms(self):
        """The POVM projectors

        Returns:
            dict: A dictionary of the POVM operators for each qubit
        """
        return self._povms

    @property
    def param_array(self):
        """The parameters for the POVMs

        Returns:
            array: a 8 * num_qubit array containing the parameters of the qubits
            in sequence
        """
        return np.array(list(self._povm_params.values())).flatten()

    def _set_povm(self):
        self._unitaries = {
            i: self._unitary_from_params(p) for i, p in self._povm_params.items()
        }
        self._povms = {
            i: self._povm_from_unitary(u) for i, u in self._unitaries.items()
        }
        self._b_matrices = {
            i: self._paulis_from_povm(p) for i, p in self._povms.items()
        }

    # pylint: disable=arguments-differ
    def construct_evaluation_circuit(
        self, wave_function, statevector_mode, q=None, c=None
    ):
        r"""
        Construct the circuit for evaluation.

        Args:
            wave_function (QuantumCircuit): the quantum circuit.
            statevector_mode (bool): indicate which type of simulator are going to use.
            qr (QuantumRegister, optional): the quantum register associated with the input_circuit
            cr (ClassicalRegister, optional): the classical register associated
                                              with the input_circuit

        Returns:
            QuantumCircuit: the quantum circuit with the POVM

        Raises:
            AquaError: if Operator is empty
            AquaError: if quantum register is not provided explicitly and
                       cannot find quantum register with `q` as the name
            AquaError: The provided qr is not in the wave_function
        """
        if self.is_empty():
            raise AquaError("Operator is empty, check the operator.")
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if statevector_mode:
            raise NotImplementedError

        if q is None:
            q = find_regs_by_name(wave_function, "q")
            if q is None:
                raise AquaError(
                    "Either provide the quantum register (qr) explicitly or use"
                    " `q` as the name of the quantum register in the input circuit."
                )
        else:
            if not wave_function.has_register(q):
                raise AquaError(
                    "The provided QuantumRegister (qr) is not in the circuit."
                )

        nq = len(q)

        if nq != self._num_qubits:
            raise AquaError(
                "The number of qubits of the provided circuit does not much that of the POVM."
            )

        if c is None:
            c = ClassicalRegister(2 * nq)

        circuit = wave_function.copy()
        qa = QuantumRegister(nq, "a")

        circuit.add_register(qa, c)

        circuit.barrier()

        ancilla = {}
        for i in range(nq):
            ancilla[i] = i

        ids = sorted(ancilla)

        # Add the POVM unitaries
        for qi in ancilla:
            circuit.unitary(self._unitaries[qi], [q[qi], qa[ancilla[qi]]])

        for i, qi in enumerate(ids):
            circuit.measure(q[qi], c[2 * i])
            circuit.measure(qa[ancilla[qi]], c[2 * i + 1])

        return circuit

    def _b_matrix_prod(self, k, m, b_matrix=None):
        if b_matrix is None:
            b_matrix = self._b_matrices

        b_prod = 1.0

        for i, ki in enumerate(k):
            # Notice that qiskit starts indexing from the right
            idx = self._num_qubits - i - 1
            i_mi = int(m[i])
            b_prod *= b_matrix[idx][ki][i_mi]
        return b_prod

    def _b_matrix_prod_array(self, bit_string):
        return np.array([self._b_matrix_prod(ki, bit_string) for ki in self._k])

    def evaluate_with_result(self, result, statevector_mode):
        """
        This method can be only used with the circuits generated by the
        :meth:`construct_evaluation_circuit` method with the same `circuit_name_prefix`
        name since the circuit names are tied to some meanings.
        Calculate the evaluated value with the measurement results.

        Args:
            result (qiskit.Result): the result from the backend.
            statevector_mode (bool): indicate which type of simulator are used.
        Returns:
            float: the mean value
            float: the standard error on the mean
        Raises:
            AquaError: if Operator is empty
        """
        if self.is_empty():
            raise AquaError("Operator is empty, check the operator.")

        if statevector_mode:
            raise NotImplementedError

        start = time()
        qiskit_counts = result.get_counts()
        sim_counts = {self._simplify(k): v for k, v in qiskit_counts.items()}

        tot_counts = np.sum(list(sim_counts.values()))

        logger.debug(
            "Computing the expectation from %d measurement results.", tot_counts
        )

        fm, sm = self.estimate_moments(sim_counts)
        ste = np.sqrt((sm - fm ** 2) / tot_counts)

        tot_time = time() - start
        logger.debug("Done in %.3f s.", tot_time)

        return fm, ste

    def estimate_moments(self, simplified_counts):
        """
        Estimate first and second moments from counts

        Args:
            simplified_counts (dict): simplified counts

        Returns:
            first_moment (float), second_moment (float)
        """
        tot_counts = np.sum(list(simplified_counts.values()))

        first_moment = 0.0
        second_moment = 0.0

        for bitstr in simplified_counts:
            b_array = self._b_matrix_prod_array(bitstr)

            first_moment += simplified_counts[bitstr] * np.dot(self._c_k, b_array)
            second_moment += simplified_counts[bitstr] * np.dot(self._c_k, b_array) ** 2

        first_moment = np.real(first_moment / tot_counts)
        second_moment = np.real(second_moment / tot_counts)

        return first_moment, second_moment

    def estimate_variance_gradient(self, result, increment):
        """Estimate the gradient of the variance of the POVM, given a result
        from a backend and an increment

        Args:
            result (qiskit.Result): the result from the backend
            increment (float): a small increment to compute the finite derivative

        Returns:
            array: the gradient of the variance of the estimator with respect to the
                   POVM parameters
        """
        qiskit_counts = result.get_counts()
        simplified_counts = {self._simplify(k): v for k, v in qiskit_counts.items()}
        tot_counts = np.sum(list(simplified_counts.values()))

        gradient = np.zeros_like(self.param_array)
        mean, ste = self.evaluate_with_result(result, False)
        std = ste * np.sqrt(tot_counts)  # We need the std of the counts, not the ste

        # with parallel_backend('threading', n_jobs=1):
        gradient = Parallel(n_jobs=4)(
            delayed(self._estimate_variance_derivative)(simplified_counts, increment, i)
            for i in range(len(self.param_array))
        )
        gradient = np.array(gradient)

        # for i in range(len(self.param_array)):
        #     gradient[i] = self._estimate_variance_derivative(
        #         simplified_counts, increment, i
        #     )

        gradient = (gradient - (std ** 2 + mean ** 2)) / increment
        return gradient

    def _estimate_variance_derivative(self, simplified_counts, increment, i):
        start = time()
        tot_counts = np.sum(list(simplified_counts.values()))
        affected_qubit = int(float(i) / 8)
        affected_param = i % 8

        target_params = self._povm_params[affected_qubit].copy()
        target_params[affected_param] += increment

        target_unitary = self._unitary_from_params(target_params)

        target_povm = self._povm_from_unitary(target_unitary)

        target_b_matrix = self._paulis_from_povm(target_povm)

        target_b_matrices = deepcopy(self._b_matrices)
        target_b_matrices[affected_qubit] = target_b_matrix

        d_matrix = self._convert_povm(self._povms[affected_qubit], target_povm)

        second_moment = 0.0

        for sample in simplified_counts:
            affected_pos = self.num_qubits - affected_qubit - 1

            b_array = self._b_matrix_prod_array(sample).copy()

            for j in range(len(self._k)):
                # If the old b_array element is non-zero, we use a speed-up,
                # otherwise we need to calculate it from scratch
                old_coeff = self._b_matrices[affected_qubit][self._k[j][affected_pos]][
                    int(sample[affected_pos])
                ]
                if b_array[j] == 0.0 or old_coeff == 0.0:
                    b_array[j] = np.nan
                else:
                    b_array[j] /= old_coeff

            b_nan = np.isnan(b_array)

            for new_outcome in range(4):
                new_sample = (
                    sample[:affected_pos]
                    + str(new_outcome)
                    + sample[affected_pos + 1 :]
                )

                new_count = (
                    d_matrix[new_outcome, int(sample[affected_pos])]
                    * simplified_counts[sample]
                )

                # We skip outcomes for which the new counts are zero
                if np.abs(new_count) > 1e-10:
                    b_array_new = b_array.copy()
                    for j in range(len(self._k)):
                        if b_nan[j]:
                            b_array_new[j] = self._b_matrix_prod(
                                self._k[j], new_sample, b_matrix=target_b_matrices
                            )
                        else:
                            b_array_new[j] *= target_b_matrix[self._k[j][affected_pos]][
                                new_outcome
                            ]

                    second_moment += new_count * np.dot(self._c_k, b_array_new) ** 2

        tot_time = time() - start
        logging.debug("Derivative of param %d calculated in %.3f seconds.", i, tot_time)
        return np.real(second_moment / tot_counts)

    @staticmethod
    def _simplify(outcome):
        n = len(outcome)
        if n % 2 != 0:
            raise AquaError("The result should have an even number of qubits.")
        simplified_outcome = ""
        for i in range(0, n, 2):
            simplified_outcome += str(2 * int(outcome[i]) + int(outcome[i + 1]))
        return simplified_outcome

    @staticmethod
    def _components_from_hypercoords(input_coords):
        """Transform a (D-1,) vector of elements in [0, 1] into a vector of
        coordinates in R^D

        Args:
            input_coords (array): a vector of parameters in [0, 1]

        Raises:
            AquaError: An error if the resulting vector is unnormalized

        Returns:
            array: a D-dimensional vector of components
        """

        hypercoords = np.pi * input_coords
        hypercoords[-1] *= 2
        rn = np.zeros(len(hypercoords) + 1)
        s = 1.0
        for i, hc in enumerate(hypercoords):
            rn[i] = s * np.cos(hc)
            s *= np.sin(hc)
        rn[-1] = s

        if not np.isclose(np.sum(rn ** 2), 1.0):
            raise AquaError(
                f"Unnormalised vector for input coordinates {input_coords}. Norm: {np.sum(rn**2)}. Vector: {rn}."
            )

        return rn

    @staticmethod
    def _unitary_from_params(params):
        """Two-qubit unitary matrix from a 8-vector

        Args:
            params (array): a 8 vector of numbers between 0 and 1

        Returns:
            array: a two-qubit unitary matrix

        Reference:
            Procedure is similar to https://quantumcomputing.stackexchange.com/questions/12789/how-do-you-embed-a-povm-matrix-in-a-unitary
        """
        # Unitary
        U = np.zeros((4, 4), dtype=complex)

        ## First column
        U[:, 0] = POVMOperator._components_from_hypercoords(params[:3])

        ## Second column
        P = np.eye(4) - np.outer(U[:, 0], U[:, 0].conj())
        orth_coords = POVMOperator._components_from_hypercoords(params[3:])
        dim = 0
        for b_vect in np.eye(4):
            proj = P @ b_vect
            if not np.isclose(proj, 0.0).all():
                proj /= np.linalg.norm(proj)
                U[:, 1] += proj * (orth_coords[2 * dim] + 1j * orth_coords[2 * dim + 1])
                P -= np.outer(proj, proj.conj())
                dim += 1
            if dim == 3:
                break

        ## Remaining two columns
        P = (
            np.eye(4)
            - np.outer(U[:, 0], U[:, 0].conj())
            - np.outer(U[:, 1], U[:, 1].conj())
        )
        dim = 0
        for b_vect in np.eye(4):
            proj = P @ b_vect
            if not np.isclose(proj, 0.0).all():
                proj /= np.linalg.norm(proj)
                U[:, dim + 2] = proj
                P -= np.outer(proj, proj.conj())
                dim += 1
            if dim == 2:
                break

        if not np.isclose(U.conj().T @ U, np.eye(4)).all():
            raise AquaError(f"U is not unitary. Params: {params}. Unitary {U}")

        return U

    @staticmethod
    def _povm_from_unitary(U):
        """Returns the 1-qubit POVM projectors from a 2-qubit unitary

        Args:
            U (array): A two-qubit unitary

        Returns:
            array: a (4, 2, 2) array with four (2, 2) projectors
        """
        povm = np.zeros((4, 2, 2), dtype=complex)
        for p in range(4):
            # The column of U is the bra of the direction of each projector
            povm[p] = np.outer(U[p, 0:2].conj(), U[p, 0:2])
        if not np.isclose(np.sum(povm, axis=0), np.eye(2)).all():
            raise AquaError("Invalid POVM unitary")
        return povm

    @staticmethod
    def _paulis_from_povm(povm):
        """Returns the decomposition of Pauli matrix for the given POVM

        Args:
            povm (array): a (4, 2, 2) array of the POVM projectors

        Returns:
            array: a (4, 4)
        """

        try:

            non_zero_povm = [
                i
                for i, p in enumerate(povm)
                if not np.isclose(p, np.zeros((2, 2))).all()
            ]
            r = len(non_zero_povm)

            # TODO: Understand if this should be an error or a warning
            if r < 4:
                print(r)

            A = np.zeros((r, r))
            for i, ii in enumerate(non_zero_povm):
                for j, ij in enumerate(non_zero_povm):
                    A[i, j] = np.real(np.trace(povm[ii] @ povm[ij]))

            decomp = {}

            for pauli in _PAULI_MATRICES:
                B = np.zeros(r)
                for i, ii in enumerate(non_zero_povm):
                    B[i] = np.real(np.trace(_PAULI_MATRICES[pauli] @ povm[ii]))
                d = np.linalg.solve(A, B)
                decomp[pauli] = np.zeros(4)
                for i, ii in enumerate(non_zero_povm):
                    decomp[pauli][ii] = d[i]
            return decomp

        except Exception as ex:
            print("Problematic POVM:", povm)
            raise (ex)

    @staticmethod
    def _convert_povm(source_povm, target_povm):
        """
        Get decomposition of target_povm in terms of source_povm

        Returns:
            Array of size (4,4)
        """
        non_zero_povm = [
            i
            for i, p in enumerate(source_povm)
            if not np.isclose(p, np.zeros((2, 2))).all()
        ]
        r = len(non_zero_povm)
        if r < 4:
            print(r)
        A = np.zeros((r, r))
        for i, ii in enumerate(non_zero_povm):
            for j, ij in enumerate(non_zero_povm):
                A[i, j] = np.real(np.trace(source_povm[ii] @ source_povm[ij]))

        decomp = np.zeros((4, 4))
        for it in range(len(target_povm)):
            B = np.zeros(r)
            for i, ii in enumerate(non_zero_povm):
                B[i] = np.real(np.trace(target_povm[it] @ source_povm[ii]))
            d = np.linalg.solve(A, B)
            for i, ii in enumerate(non_zero_povm):
                decomp[it][ii] = d[i]

        return decomp
