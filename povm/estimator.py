import numpy as np
from qiskit import execute, Aer, QuantumCircuit

from povm.povm_operator import POVMOperator
from povm.povm_optimizer import GradientDescentOptimizer

from qiskit.aqua.operators.legacy import (
    LegacyBaseOperator,
    WeightedPauliOperator,
    op_converter,
)
from qiskit.aqua.operators.legacy import TPBGroupedWeightedPauliOperator
from time import time
import logging

logger = logging.getLogger(__name__)


class Estimator:
    """
    Expectation value estimator base class
    """

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
    ):
        """Instantiates a Estimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
            return_counts (bool, optional): if True, return all the counts. Defaults to False
        Raises:
            ValueError: if the qubitOp format is not right
        """
        self.op = None

        if isinstance(qubitOp, LegacyBaseOperator):
            self.op = qubitOp
        else:
            raise ValueError("Wrong qubitOp format")

        self.backend = backend
        self.circs = qubitOp.construct_evaluation_circuit(qc, False)

        if isinstance(self.circs, QuantumCircuit):
            self.circs = [self.circs]

        self.exact = exact
        self.method = None
        self.return_counts = return_counts

    @property
    def num_circuits(self):
        """The number of measurement circuits

        Returns:
            int: Number of measurement circuits
        """
        return len(self.circs)

    def estimate(self, total_shots):
        """Estimates the expectation value using the given amount of shots for
        each measurement circuit

        Args:
            shots_per_meas (int): number of measurement shots for each measurement circuit
        Raises:
            ValueError: shots_per_meas is smaller than 1
        Returns:
            (tuple): containing
                float: expectation value
                int: effective number of shots used
        """
        shots_per_circuit = int(np.floor(total_shots / self.num_circuits))

        if shots_per_circuit < 1:
            raise ValueError(
                f"Need at least one shot per circuit. Got {total_shots} shots for {self.num_circuits} circuits."
            )

        total_shots = shots_per_circuit * self.num_circuits

        avg = 0.0

        if shots_per_circuit < 1:
            raise ValueError("There should at least be a shot per Pauli string")

        start = time()
        result = execute(
            self.circs,
            backend=Aer.get_backend("qasm_simulator"),
            shots=shots_per_circuit,
        ).result()
        qc_time = time() - start

        start = time()
        avg, std = np.real(self.op.evaluate_with_result(result, False))

        post_processing_time = time() - start
        logger.info(
            "Circuit time %.2f s. Postproc. time %.2f s.", qc_time, post_processing_time
        )

        yield {
            "qubits": self.op.num_qubits,
            "true": self.exact,
            "estimate": avg,
            "estimated_error": std,
            "error": np.abs(avg - self.exact),
            "circuits": self.num_circuits,
            "shots_per_circuit": shots_per_circuit,
            "shots": total_shots,
            "time_qc": qc_time,
            "time_post": post_processing_time,
            "method": self.method,
            "counts": result.get_counts() if self.return_counts else None,
        }


class PauliEstimator(Estimator):
    """
    Expectation value estimator using Weighted Pauli strings
    """

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
    ):
        """Instantiates a PauliEstimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc.
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
        Raises:
            ValueError: if the qubitOp format is not right
        """
        if isinstance(qubitOp, WeightedPauliOperator):
            super().__init__(
                qc, qubitOp, exact, backend=backend, return_counts=return_counts
            )
        else:
            raise ValueError("Expected a WeightedPauliOperator")

        self.method = "Pauli"


class GroupedPauliEstimator(Estimator):
    """
    Expectation value estimator using Weighted Pauli strings
    """

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
    ):
        """Instantiates a PauliEstimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc.
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
        Raises:
            ValueError: if the qubitOp format is not right
        """
        if isinstance(qubitOp, GroupedPauliEstimator):
            super().__init__(
                qc, qubitOp, exact, backend=backend, return_counts=return_counts
            )
        elif isinstance(qubitOp, WeightedPauliOperator):
            op = op_converter.to_tpb_grouped_weighted_pauli_operator(
                qubitOp, TPBGroupedWeightedPauliOperator.sorted_grouping
            )
            super().__init__(
                qc, op, exact, backend=backend, return_counts=return_counts
            )
        else:
            raise ValueError(
                "Expected a GroupedPauliEstimator or WeightedPauliOperator"
            )

        self.method = "Grouped_Pauli"


class POVMEstimator(Estimator):
    """
    Expectation value estimator using Weighted Pauli strings
    """

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
        povm_params=None
    ):
        """Instantiates a PauliEstimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc.
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
        Raises:
            ValueError: if the qubitOp format is not right
        """
        if isinstance(qubitOp, POVMOperator):
            super().__init__(
                qc, qubitOp, exact, backend=backend, return_counts=return_counts
            )
        elif isinstance(qubitOp, WeightedPauliOperator):
            op = POVMOperator(qubitOp, povm_params=povm_params)
            super().__init__(
                qc, op, exact, backend=backend, return_counts=return_counts
            )
        else:
            raise ValueError("Expected a POVMOperator or WeightedPauliOperator")

        self.method = "SIC-POVM"

    def estimate(self, total_shots):
        res = super().estimate(total_shots)

        for item in res:
            item["povm_params"] = list(self.op.param_array)
            yield item


class GooglePOVMEstimator(Estimator):
    """
    Expectation value estimator using Weighted Pauli strings
    """

    _Google_params = [
        0.2838889,
        0.36284288,
        0.57602168,
        0.40456533,
        0.3999603,
        0.28235376,
        0.6941424,
        0.33333333,
    ]

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
    ):
        """Instantiates a PauliEstimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc.
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
        Raises:
            ValueError: if the qubitOp format is not right
        """
        if isinstance(qubitOp, POVMOperator):
            qubitOp = POVMOperator(
                qubitOp, povm_params=self._Google_params * qubitOp.num_qubits
            )
            super().__init__(
                qc, qubitOp, exact, backend=backend, return_counts=return_counts
            )
        elif isinstance(qubitOp, WeightedPauliOperator):
            op = POVMOperator(
                qubitOp, povm_params=self._Google_params * qubitOp.num_qubits
            )
            super().__init__(
                qc, op, exact, backend=backend, return_counts=return_counts
            )
        else:
            raise ValueError("Expected a POVMOperator or WeightedPauliOperator")

        self.method = "Google-POVM"

    def estimate(self, total_shots):
        res = super().estimate(total_shots)

        for item in res:
            item["povm_params"] = list(self.op.param_array)
            yield item


class GradPOVMEstimator(Estimator):
    """
    Expectation value estimator using Weighted Pauli strings
    """

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
    ):
        """Instantiates a PauliEstimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc.
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
        Raises:
            ValueError: if the qubitOp format is not right
        """
        if isinstance(qubitOp, POVMOperator):
            super().__init__(
                qc, qubitOp, exact, backend=backend, return_counts=return_counts
            )
        elif isinstance(qubitOp, WeightedPauliOperator):
            op = POVMOperator(qubitOp)
            super().__init__(
                qc, op, exact, backend=backend, return_counts=return_counts
            )
        else:
            raise ValueError("Expected a POVMOperator or WeightedPauliOperator")

        # We need to overwrite the base circuit construction
        self.circs = [qc]
        self.method = "Grad-POVM"
        self.optimizer = None

    def estimate(self, total_shots):
        self.optimizer = GradientDescentOptimizer(
            self.circs[0],
            self.op,
            total_shots,
            initial_shots=1000,
            shot_increment=1000,
            nu_factor=1.2,
            exact_value=self.exact,
        )

        result = self.optimizer.run_step(return_counts=self.return_counts)

        while result is not None:
            result["method"] = self.method
            yield result
            result = self.optimizer.run_step(return_counts=self.return_counts)


class GoogleGradPOVMEstimator(Estimator):
    """
    Expectation value estimator using Weighted Pauli strings
    """

    _Google_params = [
        0.2838889,
        0.36284288,
        0.57602168,
        0.40456533,
        0.3999603,
        0.28235376,
        0.6941424,
        0.33333333,
    ]

    def __init__(
        self,
        qc,
        qubitOp,
        exact,
        backend=Aer.get_backend("qasm_simulator"),
        return_counts=False,
    ):
        """Instantiates a PauliEstimator
        Args:
            qc (QuantumCircuit): circuit implementing the state
            qubitOp (WeightedPauliOperator): The operator to evaluate
            exact (float): The exact value for the operator on the circuit qc.
            backend (BaseBackend, optional): the backend for running the circuit.
                                Defaults to Aer.get_backend('qasm_simulator').
        Raises:
            ValueError: if the qubitOp format is not right
        """
        if isinstance(qubitOp, POVMOperator):
            super().__init__(
                qc,
                POVMOperator(
                    qubitOp, povm_params=self._Google_params * qubitOp.num_qubits
                ),
                exact,
                backend=backend,
                return_counts=return_counts,
            )
        elif isinstance(qubitOp, WeightedPauliOperator):
            op = POVMOperator(
                qubitOp, povm_params=self._Google_params * qubitOp.num_qubits
            )
            super().__init__(
                qc, op, exact, backend=backend, return_counts=return_counts
            )
        else:
            raise ValueError("Expected a POVMOperator or WeightedPauliOperator")

        # We need to overwrite the base circuit construction
        self.circs = [qc]
        self.method = "Grad-Google-POVM"
        self.optimizer = None

    def estimate(self, total_shots):
        self.optimizer = GradientDescentOptimizer(
            self.circs[0],
            self.op,
            total_shots,
            initial_shots=1000,
            shot_increment=1000,
            nu_factor=1.2,
            exact_value=self.exact,
        )

        result = self.optimizer.run_step(return_counts=self.return_counts)
        while result is not None:
            result["method"] = self.method
            yield result
            result = self.optimizer.run_step(return_counts=self.return_counts)
