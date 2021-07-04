import numpy as np

# Log-likelihood function
def log_l(rho, observables, counts):
    return np.sum(counts * np.log(np.real(np.tensordot(observables, rho.T)))) / np.sum(
        counts
    )


# R(rho) operator
def R(rho, observables, counts):
    R = np.zeros_like(rho)

    traces = np.tensordot(observables, rho.T)
    for obs in range(len(counts)):
        if counts[obs] != 0:
            if traces[obs] != 0:
                R += (counts[obs] / traces[obs]) * observables[obs]
    R /= sum(counts)
    return R


# Returns rho^{(k+1)} given rho (not diluted)
def RrR(rho, observables, marginals):
    RR = R(rho, observables, marginals)
    rhok = np.dot(RR, np.dot(rho, RR))
    return rhok / rhok.trace()


# Returns rho^{(k+1)} given rho and epsilon
def IRrR(rho, rrho, epsilon):
    M = (np.eye(len(rho)) + epsilon * rrho) / (1 + epsilon)
    rhok = np.dot(M, np.dot(rho, M))
    return rhok / rhok.trace()


# Maximises the log-likelihood
def infer_state(
    marginals,
    qlist,
    observables,
    tol=1e-15,
    maxiter=2000,
    epsilon_range=1e-6,
    n_epsilons=50,
):
    """
    Returns the state that maximises the log-likelihood given the observations.
    input:
        qlist (tuple): qubits for which the maximisation is carried out.
        observables (dict): dictionary with the effect (numpy array) corresponding to each outcome.
        tol (float): tolerance for the convergence of the algorithm.
        maxiter (int): maximum number of iterations.
        epsilon_range (float): range in which random values of epsilon are sampled in the second and third phases.
        n_epsilons (int): number of values of epsilon in the range (0, epsilon_range] to be maximised over in phase 3.

        The format for the keys in 'marginals' and 'observables' is a chain of outcomes for the given POVM with
        the same order as qlist. For instance, '031' corresponds to qlist[0] with outcome '1', qlist[1] yielding '3',
        and qlist[2], '0'.
    output:
        A density matrix (numpy array).
    """
    k = len(qlist)
    counts = np.array(list(marginals[qlist].values()))
    observables = np.array(list(observables.values()))

    # Phase 1: iterative algorithm without (not diluted)
    rhok = np.eye(2 ** k) / 2 ** k + 0j
    logl_rhok = log_l(rhok, observables, counts)

    for iteration_one in range(maxiter):
        rho = rhok
        logl_rho = logl_rhok

        rhok = RrR(rho, observables, counts)
        logl_rhok = log_l(rhok, observables, counts)
        if logl_rhok < logl_rho:
            # Stop if likelihood decreases (and do not accept last step)
            rhok = rho
            logl_rhok = logl_rho
            break
        elif (
            np.isclose(
                logl_rhok,
                logl_rho,
                atol=tol,
                rtol=0,
            )
            and np.allclose(rhok, rho, atol=tol, rtol=0)
        ):
            # Stop if increase in likelihood and rhok-rho are small enough
            break

    # Phase 2: iterate diluted algorithm with random epsilon
    for iteration_two in range(maxiter):
        rho = rhok
        logl_rho = logl_rhok

        epsilon = np.random.rand() * epsilon_range
        rrho = R(rho, observables, counts)
        rhok = IRrR(rho, rrho, epsilon)

        logl_rhok = log_l(rhok, observables, counts)

        if logl_rhok < logl_rho:
            # If likelihood decreases, do not accept the change but continue
            rhok = rho
            logl_rhok = logl_rho
        elif (
            np.isclose(
                logl_rhok,
                logl_rho,
                atol=tol,
                rtol=0,
            )
            and np.allclose(rhok, rho, atol=tol, rtol=0)
        ):
            # Stop if increase in likelihood and rhok-rho are small enough
            break

    # Phase 3: iterate dilute algorithm for largest value of epsilon
    epsilons = np.linspace(0, epsilon_range, n_epsilons + 1)[1:]
    for iteration_three in range(maxiter):
        # Find largest increase in log-likelihood
        rrhok = R(rhok, observables, counts)
        delta_logl = {
            epsilon: log_l(IRrR(rhok, rrhok, epsilon), observables, counts) - logl_rhok
            for epsilon in epsilons
        }
        max_epsilon = max(delta_logl, key=delta_logl.get)
        if delta_logl[max_epsilon] > tol:
            rhok = IRrR(rhok, rrhok, epsilon)
            logl_rhok = log_l(rhok, observables, counts)
        else:
            break

    # Verify result
    # rrhok = R(rhok, observables, counts)
    # delta_logl = {
    #     epsilon: log_l(IRrR(rhok, rrhok, epsilon), observables, counts) - logl_rhok
    #     for epsilon in epsilons
    # }

    if not (
        max(delta_logl.values()) < tol
        and np.isclose(
            logl_rhok,
            logl_rho,
            atol=tol,
            rtol=0,
        )
        and np.allclose(rhok, rho, atol=tol, rtol=0)
    ):
        print("Convergence not achieved:")
        print(
            "Delta log-likelihood:",
            np.abs(logl_rhok - logl_rho),
        )
        print("Largest difference in operators:", np.amax(np.abs(rho - rhok)))
        print("Iterations:")
        print("Phase 1:", iteration_one + 1)
        print("Phase 2:", iteration_two + 1)
        print("Phase 3:", iteration_three + 1)

    return rhok
