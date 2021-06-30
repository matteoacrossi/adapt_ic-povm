""" A Montecarlo POVM optimizer """
import numpy as np
import logging

from .povm_operator import POVMOperator

from qiskit import execute, Aer
from qiskit.aqua import AquaError

import cma
from joblib import Parallel, delayed

from time import time

logger = logging.getLogger(__name__)


class POVMResampler:
    """Classical estimator of performance of a POVM based on outcomes of another POVM"""

    def __init__(self, source_povm: POVMOperator, target_povm: POVMOperator):
        """
        Args:
            source_povm (POVMOperator): POVM outcomes of which to be sampled from
            target_povm (POVMOperator): POVM performance of which to be estimated
        """
        self._source = source_povm
        self._target = target_povm

        # Calculate the transformation matrices between POVMs
        self._c_matrices = np.array(
            [
                self._convert_povm(self._source.povms[i], self._target.povms[i])
                for i in range(self.num_qubits)
            ]
        )

    @property
    def num_qubits(self):
        """The number of qubits on which the POVM is performed

        Returns:
            int: the number of qubits on which the POVM is performed
        """
        return self._source._num_qubits

    def resample(self, result, shots_per_sample):
        """
        Monte Carlo sample a target_povm based on results obtained with source_povm

        Args:
            result (qiskit.Result): the result from the backend obtained using source_povm
            shots_per_sample (int): number of Monte Carlo shots per real outcome sample

        Returns:
            sm (float): estimated second moment of target_povm
        """
        qiskit_counts = result.get_counts()
        input_samples = {POVMOperator._simplify(k): v for k, v in qiskit_counts.items()}

        # Some quantities depending on c_matrices
        abs_c_matrices = np.abs(self._c_matrices)
        sum_abs_c_matrices = np.sum(abs_c_matrices, axis=1)
        probs_mat = abs_c_matrices / sum_abs_c_matrices[:, np.newaxis, :]

        # Simulate outcomes with target POVM based on source POVM
        samples = {}
        for sample, counts in input_samples.items():
            for _ in range(shots_per_sample * counts):
                simulated_outcome = ""
                beta = 1.0
                for q in range(len(sample)):
                    outcome = int(sample[self.num_qubits - 1 - q])
                    # Outcome on qubit q decided via importance sampling
                    so = self._fast_sampler(probs_mat[q, :, outcome])
                    simulated_outcome += str(so)
                    # beta is the weighted contribution to the integral
                    # corrected by the sample probability
                    beta *= (
                        np.sign(self._c_matrices[q, so, outcome])
                        * sum_abs_c_matrices[q, outcome]
                    )

                # Reverse order of string to make Qiskit-compatible
                simulated_outcome = "".join(
                    [
                        simulated_outcome[self.num_qubits - 1 - a]
                        for a in range(len(simulated_outcome))
                    ]
                )

                # Store results in self.samples as a dictionary containing beta for each simulated outcome
                if simulated_outcome not in samples:
                    samples[simulated_outcome] = 0
                samples[simulated_outcome] += beta

        _, sm = self._target.estimate_moments(samples)
        return sm

    @staticmethod
    def _fast_sampler(probs):
        """Faster than np.random.choice"""
        s_index = -1
        ps = 0.0
        p = np.random.rand()
        while ps < p:
            s_index += 1
            ps += probs[s_index]
        return s_index

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


class POVMOptimizer:
    """
    Classically optimize a n-qubit POVM to reduce the estimation variance, given a quantum
    circuit
    """

    def __init__(self, povm_op, result, shots_per_sample=1):
        """
        Args:
            povm_op (POVMOperator): A POVMOperator object to be optimized
            result (qiskit.Result): the result from the backend obtained using existing POVM
            shots_per_sample (int): number of Monte Carlo shots per real outcome sample
        """
        self.povm_op = povm_op
        self.result = result
        self.shots_per_sample = shots_per_sample
        self._max_attempts = 100
        self._optimal_params = None

        # Store the mean from the initial estimation
        self._mean, _ = self.povm_op.evaluate_with_result(self.result, False)

    def optimize(self, tolfun=1e-5, max_iter=100, initial_var=0.02, n_jobs=None):
        r"""
        Finds a POVM with minimal variance using CMA-ES strategy.
        The minimization is parallelized using joblib.
        The optimal POVMOperator can be retrieved with
        `POVMOptimizer.optimal_povm`.

        Args:
            tolfun (float, optional): Tolerance of the minimization. Defaults to 1e-5.
            max_iter (int, optional): maximum number of iterations. Defaults to 100.
            initial_var (float, optional): initial variance for CMA. Defaults to 0.02.
            n_jobs (int, optional): Number of parallel jobs. If -1, use all the
                        available CPUs, if None or 1, don't use parallelization.
                        Defaults to None.

        """
        es = cma.CMAEvolutionStrategy(
            self.povm_op.param_array,
            initial_var,
            dict(verbose=1, tolfun=tolfun, bounds=[0.001, 0.999]),
        )

        with Parallel(n_jobs=n_jobs) as parallel:

            for _ in range(max_iter):
                if not es.stop():
                    solutions = es.ask()
                    answers = parallel(
                        delayed(self.estimate_variance)(sol) for sol in solutions
                    )
                    # answers = [self.estimate_second_moment(sol) for sol in solutions]
                    # answers = parallel_map(self.estimate_second_moment,
                    #                       solutions,
                    #                       num_processes=aqua_globals.num_processes)
                    es.tell(solutions, answers)
                    es.disp()
                    self._optimal_params = es.best.x
                else:
                    print("Optimization converged.")

    @property
    def optimal_povm(self):
        """
        Returns:
            (POVMOperator): Most optimal POVM obtained so far
        """
        if self._optimal_params is None:
            return AquaError("Run optimize first!")
        else:
            return POVMOperator(self.povm_op.paulis, povm_params=self._optimal_params)

    def estimate_variance(self, povm_params):
        """Returns the estimated variance for the POVM defined by
        povm_params to use in the optimisation.

        Args:
            povm_params (array or list): the parameter array of the POVM

        Returns:
            float: the estimated variance of the new POVM

        Remarks:
        Given that a few-shots calculation can yield an estimated second
        moment smaller than the squared mean, we return a very large value if
        that happens.
        """
        target_povm = POVMOperator(self.povm_op.paulis, povm_params=povm_params)
        resampler = POVMResampler(self.povm_op, target_povm)

        for _ in range(self._max_attempts):
            second_moment = resampler.resample(self.result, self.shots_per_sample)

            if second_moment > self._mean ** 2:
                var = np.sqrt(second_moment - self._mean ** 2)
                return var

        # We kept getting wrong estimation of the variance, return a large number
        return 1e6


class GradientDescentOptimizer:
    def __init__(
        self,
        qc,
        initial_povm_op,
        max_shots=1e5,
        initial_shots=100,
        shot_increment=100,
        initial_nu=5e-2,
        nu_factor=1.3,
        update_frequency=3,
        increment=1e-3,
        parameter_threshold=0.005,
        exact_value=None,
        seed=None,
        backend=Aer.get_backend("qasm_simulator"),
    ):

        # Set configuration parameters
        self.qc = qc
        self.initial_povm_op = initial_povm_op
        self.max_shots = max_shots
        self.initial_shots = initial_shots
        self.shot_increment = shot_increment
        self.initial_nu = initial_nu
        self.nu_factor = nu_factor
        self.update_frequency = update_frequency
        self.increment = increment
        self.exact_value = exact_value
        self.seed = seed
        self.backend = backend
        self.parameter_threshold = parameter_threshold

        # Set state variables
        self.shots = self.initial_shots
        self.nu = self.initial_nu
        self.total_shots = 0
        self.step = 0
        self.povm_params = self.initial_povm_op.param_array.copy()
        self.est_mean = None
        self.est_var = None

    def run_step(self, return_counts=False):
        if self.total_shots >= self.max_shots:
            # TODO: maybe raise an exception
            return None

        self.step += 1

        # Update parameters if necessary
        if self.step % self.update_frequency == 0:
            self.shots += self.shot_increment
            self.nu /= self.nu_factor

        self.shots = int(min(self.shots, self.max_shots - self.total_shots))
        self.total_shots += self.shots

        # Get data from quantum computer
        povm_op = POVMOperator(self.initial_povm_op, povm_params=self.povm_params)
        povm_qc = povm_op.construct_evaluation_circuit(self.qc, False)

        if self.step == 1:
            logger.info("Step\tShots\tstep_size\tMean\tSte\tError\tTime Q\tTime C")
            logger.info(
                "--------------------------------------------------------------------------"
            )

        logger.debug("Operator created")
        start = time()
        result = execute(
            povm_qc,
            backend=self.backend,
            seed_simulator=self.seed + self.step if self.seed is not None else None,
            shots=self.shots,
        ).result()

        elapsed_qc = time() - start

        logger.debug("Quantum circuit run in %.2f", elapsed_qc)
        start = time()

        # Evaluate sample mean and ste from the counts
        sample_mean, sample_mean_ste = povm_op.evaluate_with_result(result, False)

        logger.debug("Result evaluated")

        # Update the estimator with the new estimates
        sample_est_var = sample_mean_ste ** 2

        if self.step == 1:
            self.est_mean = sample_mean
            self.est_var = sample_mean_ste ** 2
        else:
            alpha = self.est_var / (self.est_var + sample_est_var)
            self.est_mean = alpha * sample_mean + (1.0 - alpha) * self.est_mean
            self.est_var = (
                alpha ** 2 * sample_est_var + (1.0 - alpha) ** 2 * self.est_var
            )

        # er = f"{np.abs(est_mean - exact_value):.3f}" if self.exact_value else "-"

        # Calculate gradient
        grad = povm_op.estimate_variance_gradient(result, self.increment)

        elapsed_grad = time() - start

        logger.debug("Gradient calculated in %.2f", elapsed_grad)
        # Perform the gradient descent
        # The parameters are clipped between 0 and 1
        self.old_povm_params = self.povm_params
        self.povm_params = np.clip(
            self.povm_params - self.nu * grad / np.max(np.abs(grad)),
            0.0 + self.parameter_threshold,
            1.0 - self.parameter_threshold,
        )

        step_result = {
            "qubits": self.initial_povm_op.num_qubits,
            "true": self.exact_value,
            "estimate": self.est_mean,
            "estimated_error": np.sqrt(self.est_var),
            "error": np.abs(self.est_mean - self.exact_value)
            if self.exact_value
            else None,
            "nu": self.nu,
            "step": self.step,
            "circuits": 1,
            "shots_per_circuit": self.total_shots,
            "shots": self.total_shots,
            "povm_params": list(self.old_povm_params),
            "time_qc": elapsed_qc,
            "time_post": elapsed_grad,
        }

        logger.info(
            "%3d\t%6d\t%.3f\t\t%.3f\t%.3f\t%s\t%.1f\t%.1f",
            step_result["step"],
            step_result["shots"],
            step_result["nu"],
            step_result["estimate"],
            step_result["estimated_error"],
            step_result["error"],
            step_result["time_qc"],
            step_result["time_post"],
        )

        if return_counts:
            step_result["counts"] = result.get_counts()

        return step_result


def gradient_descent_optimize(
    qc,
    initial_povm_op,
    max_shots=1e5,
    initial_shots=100,
    shot_increment=100,
    initial_nu=5e-2,
    nu_factor=1.3,
    update_frequency=3,
    increment=1e-3,
    exact_value=None,
    seed=None,
    backend=Aer.get_backend("qasm_simulator"),
):
    """Iteratively optimize the POVM measurement for estimating the expectation
    value of given operator on a quantum circuit, using a gradient descent
    method.
    At each iteration, shots are collected from a backend. All the shots are
    then used to build an estimation for the expectation value of the operator.

    The function uses an exponentially decreasing schedule for the step size nu.
    And a linearly increasing schedule for the number of shots.

    Args:
        qc (qiskit.QuantumCircuit): the quantum circuit on which to evaluate the
                expectation value of the operator
        initial_povm_op (POVMOperator): the POVMOperator to optimize
        max_shots (int, optional): Total number of shots to generate on the backend. Defaults to 1e5.
        initial_shots (int, optional): Shots for the initial evaluation. Defaults to 100.
        initial_nu (float, optional): Initial gradient step size. Defaults to 5e-2.
        shot_increment (int, optional): Increment in the sampling shots. Defaults to 100.
        nu_factor (float, optional): Exponential factor for the step size. Defaults to 1.3.
        update_frequency (int, optional): how many steps between updates of nu and shots. Defaults to 3.
        exact_value (float, optional): if given, it is used to evaluate the error of the estimation. Defaults to None.
        seed (int, optional): seed for the sampling. Defaults to None.
        backend (qiskit.providers.BaseBackend, optional): A backend. Defaults to Aer.get_backend('qasm_simulator').
        increment (float, optional): a small increment to numerically evaluate the gradient. Defaults to 1e-3.

    Returns:
        GradientOptimizationResult: the result of the optimization
    """

    # Variable initialisation
    shots = initial_shots
    nu = initial_nu
    total_shots = 0
    step = 0

    current_povm_params = initial_povm_op.param_array.copy()

    opt_result = GradientOptimizationResult(
        initial_op=initial_povm_op,
        param_seq=[],
        shot_seq=[],
        mean_seq=[],
        ste_seq=[],
        gradient_seq=[],
        optimal_povm_params=None,
    )

    opt_result.param_seq.append(current_povm_params.copy())

    logger.info("Step\tShots\tstep_size\tMean\tSte\tError\tTime Q\tTime C")
    logger.info(
        "--------------------------------------------------------------------------"
    )

    while total_shots < max_shots:
        step += 1
        # Update parameters if necessary
        if step % update_frequency == 0:
            shots += shot_increment
            nu /= nu_factor

        shots = int(min(shots, max_shots - total_shots))
        total_shots += shots

        # Get data from quantum computer
        povm_op = POVMOperator(initial_povm_op, povm_params=current_povm_params)
        povm_qc = povm_op.construct_evaluation_circuit(qc, False)

        start = time()
        result = execute(
            povm_qc,
            backend=backend,
            seed_simulator=seed + step if seed is not None else None,
            shots=shots,
        ).result()
        elapsed_qc = time() - start

        start = time()
        # Evaluate sample mean and ste from the counts
        sample_mean, sample_mean_ste = povm_op.evaluate_with_result(result, False)

        # Update the estimator with the new estimates
        sample_est_var = sample_mean_ste ** 2
        if step == 1:
            est_mean = sample_mean
            est_var = sample_mean_ste ** 2
        else:
            alpha = est_var / (est_var + sample_est_var)
            est_mean = alpha * sample_mean + (1.0 - alpha) * est_mean
            est_var = alpha ** 2 * sample_est_var + (1.0 - alpha) ** 2 * est_var

        opt_result.shot_seq.append(total_shots)
        opt_result.mean_seq.append(est_mean)
        opt_result.ste_seq.append(np.sqrt(est_var))

        er = f"{np.abs(est_mean - exact_value):.3f}" if exact_value else "-"

        # Calculate gradient
        grad = povm_op.estimate_variance_gradient(result, increment)

        elapsed_grad = time() - start
        # Perform the gradient descent
        current_povm_params -= nu * grad / np.max(np.abs(grad))

        # The parameters are clipped between 0 and 1
        np.clip(
            current_povm_params,
            0 + PARAM_THRESHOLD,
            1 - PARAM_THRESHOLD,
            out=current_povm_params,
        )

        opt_result.param_seq.append(current_povm_params.copy())
        opt_result.gradient_seq.append(grad)

        opt_result.optimal_povm_params = current_povm_params

        logger.info(
            "%3d\t%6d\t%.3f\t\t%.3f\t%.3f\t%s\t%.1f\t%.1f",
            step,
            total_shots,
            nu,
            est_mean,
            np.sqrt(est_var),
            er,
            elapsed_qc,
            elapsed_grad,
        )
    return opt_result


class GradientOptimizationResult(dict):
    """Result of a gradient_descent_optimize run, containing debug information
    on the run.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            nsteps = self.steps
            mean = self.mean_seq[-1]
            ste = self.ste_seq[-1]
            impr = self.ste_seq[0] / ste
            total_shots = self.shot_seq[-1]
            return "\n".join(
                (
                    f"Steps: {nsteps}",
                    f"Shots: {total_shots}",
                    f"Result: {mean} ({ste})",
                    f"Initial result: {self.mean_seq[0]} ({self.ste_seq[0]})",
                    f"Improvement: {impr:.2f}x",
                )
            )
            # m = max(map(len, list(self.keys()))) + 1
            # return '\n'.join([k.rjust(m) + ': ' + repr(v)
            #                   for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    @property
    def optimal_povm(self):
        """Optimal POVM Operator

        Returns:
            POVMOperator: Operator with the optimal POVM measurement
        """
        povmop = POVMOperator(self.initial_op, povm_params=self.optimal_povm_params)
        return povmop

    @property
    def steps(self):
        """Number of steps for the optimization

        Returns:
            int: number of steps
        """
        return len(self.mean_seq)

    def print_log(self, exact_value=None):
        print(f"Step\tShots\tMean\tSte\tError")
        print(f"-----------------------------------")
        for i in range(self.steps):
            er = f"{np.abs(self.mean_seq[i] - exact_value):.3f}" if exact_value else "-"
            print(
                f"{i+1}\t{self.shot_seq[i]}\t{self.mean_seq[i]:.3f}\t{self.ste_seq[i]:.3f}\t{er}"
            )

    def to_dataset(self, exact_value):
        result = []
        for i in range(self.steps):
            result.append(
                {
                    "Qubits": self.initial_op.num_qubits,
                    "True": exact_value,
                    "Estimate": self.mean_seq[i],
                    "Est. std": self.ste_seq[i],
                    "Error": np.abs(self.mean_seq[i] - exact_value),
                    "Circuits": 1,
                    "Shots per circuit": self.shot_seq[i],
                    "Shots": self.shot_seq[i],
                    "Parameters": self.param_seq[i],
                    "Method": "Grad. POVM",
                }
            )
        return result