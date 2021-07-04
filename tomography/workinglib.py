import numpy as np
from baseconvert import base
import itertools
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def label(index, s, t):
    l = base(index, 10, s, string=True)
    return "0" * (t - len(l)) + l


def create_labels_list(s, n):
    return [label(i, s, n) for i in range(s ** n)]


def generate_sicpovm_pdf(rho, n):
    # SIC-POVM rank-1 projectors
    a = {}
    a[0] = np.array([1, 0])
    a[1] = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)])
    a[2] = np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
    a[3] = np.array([1 / np.sqrt(3), np.exp(1j * 4 * np.pi / 3) * np.sqrt(2 / 3)])
    p = {}
    for i in range(4):
        p[i] = np.outer(a[i], a[i].conj()) / 2

    # Compute corresponding probabilities
    outcomes = create_labels_list(4, n)
    pdf = np.zeros(4 ** n)
    for i in range(len(outcomes)):
        effect = p[int(outcomes[i][0])]
        for j in range(1, n):
            effect = np.kron(effect, p[int(outcomes[i][j])])
        pdf[i] = np.real(np.trace(effect @ rho))
    return pdf


def generate_sicpovm_effects(n, order="Standard"):
    # SIC-POVM rank-1 projectors
    a = {}
    a[0] = np.array([1, 0])
    a[1] = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)])
    a[2] = np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
    a[3] = np.array([1 / np.sqrt(3), np.exp(1j * 4 * np.pi / 3) * np.sqrt(2 / 3)])
    p = {i: np.outer(a[i], a[i].conj()) / 2 for i in range(4)}

    obs = {}
    for ijk in create_labels_list(4, n):
        effect = p[int(ijk[0])]
        for e in ijk[1:]:
            if order == "Standard":
                effect = np.kron(p[int(e)], effect)
            else:
                effect = np.kron(effect, p[int(e)])
        obs[ijk] = effect

    return obs


def sicpovm_unitary():

    h = np.sqrt(1 / 2)
    a = np.sqrt(2 / 3)
    b = np.exp(-1j * 2 * np.pi / 3)

    v1 = np.array([1, a * h, a * h, a * h])
    v2 = np.array([0, a, a * b, a * b ** 2])
    v3 = np.array([0, a, a * b ** 2, a * b])
    v4 = np.array([-1, a * h, a * h, a * h])

    U = h * np.array([v1, v2, v3, v4]).T

    return U


# Function that adds the measurement unitaries
def sicpovm(qc, q, c, ancilla, ids=[]):
    """
    Prepares the measurement circuit.
    input:
        qc (QuantumCircuit): circuit preparing the state to be measured.
        q (QuantumRegister): qubit register including ancillae.
        c (ClassicalRegister): cbit register for outcomes.
        ancilla (dict): dictionary yielding the ancillary qubit for each logical qubit.
        ids (list): list of qubits in the same order as in the outcome string.
    ouput:
        A quantum circuit implementing the SIC-POVM on every qubit in 'ancilla'.
    """

    # Add the SIC-POVM unitaries
    for qi in ancilla:
        qc.unitary(sicpovm_unitary(), [q[qi], q[ancilla[qi]]])

    # Add the measurements
    if len(ids) == 0:
        ids = sorted(ancilla)
    for i, qi in enumerate(ids):
        qc.measure(q[qi], c[2 * i])
        qc.measure(q[ancilla[qi]], c[2 * i + 1])

    return qc


def B(q, i, j, p):
    """
    Returns the two-qubit gate that "splits" the excitation between qubits i and j with weight p
    """
    B = QuantumCircuit(q)

    theta = 2 * np.arcsin(np.sqrt(1.0 - p))

    # Bare gate
    # B.cu3(theta, 0.0, 0.0, q[i], q[j])
    # Proposal by Clément
    thetap = np.arcsin(np.cos(theta / 2.0))
    B.u3(thetap, 0.0, 0.0, q[j])
    B.cx(q[i], q[j])
    B.u3(-thetap, 0.0, 0.0, q[j])

    B.cx(q[j], q[i])

    return B


def B0(q, i, j, p):
    """
    Returns the two-qubit gate that "splits" the excitation between qubits i and j with weight p for the first pair
    """
    B0 = QuantumCircuit(q)

    theta = 2 * np.arcsin(np.sqrt(1.0 - p))

    B0.x(q[i])
    B0.u3(theta, 0.0, 0.0, q[j])
    B0.cx(q[j], q[i])

    return B0


def w_state(q, c, protocol):
    """
    Consruct the circuit with the corresponding parallelisation. The input list protocol contains a set
    of lists with the pairs of B gates to be parallelised at a given time.
    """
    qc = QuantumCircuit(q, c)

    for pair_index, pair_set in enumerate(protocol):
        for i, j, p in pair_set:
            if pair_index == 0:
                qc += B0(q, i, j, p)
            else:
                qc += B(q, i, j, p)

        # qc.barrier()

    return qc


def check_protocol(protocol):
    """
    Check if a given protocol works by computing the qubit excitation probabilities
    """
    qubit_weight = {}
    qubit_weight[protocol[0][0][0]] = 1.0
    for pair_set in protocol:
        for i, j, p in pair_set:
            qubit_weight[j] = qubit_weight[i] * (1.0 - p)
            qubit_weight[i] *= p

    return qubit_weight


def protocol_from_tree(tree):
    """
    Determine the gate paramters from a given tree
    """
    # Determine number of descendants
    descendants = {}
    for i in range(len(tree) - 1, -1, -1):
        pair_set = tree[i]
        for i, j in pair_set:
            if i not in descendants:
                descendants[i] = 1
            if j not in descendants:
                descendants[j] = 1
            descendants[i] += descendants[j]

    # Assign probabilities to edges
    protocol = []
    excitations = {}
    excitations[tree[0][0][0]] = len(descendants)
    for pair_set in tree:
        new_pair_set = []
        for i, j in pair_set:
            p = 1.0 - float(descendants[j]) / excitations[i]
            excitations[j] = descendants[j]
            excitations[i] -= excitations[j]
            new_pair_set.append((i, j, p))
        protocol.append(new_pair_set)

    return protocol


def simplify(outcome):

    n = len(outcome)
    if n % 2 != 0:
        print("Odd number of qubits")

    simplified_outcome = ""
    for i in range(0, n, 2):
        simplified_outcome += str(2 * int(outcome[i]) + int(outcome[i + 1]))

    return simplified_outcome


def marginalise_outcomes(probs, kple):

    n = len(list(probs.keys())[0])
    k = len(kple)
    marginal = {outcome: 0.0 for outcome in create_labels_list(4, k)}
    for outcome in probs:
        marginal_outcome = ""
        for i in range(k):
            marginal_outcome += outcome[n - 1 - kple[k - 1 - i]]
        marginal[marginal_outcome] += probs[outcome]

    return marginal


def compute_all_simplified_marginals(counts, ids, k):

    probs = {simplify(outcome): counts[outcome] for outcome in counts}
    marginals = {
        tuple([ids[i] for i in kple]): marginalise_outcomes(probs, kple)
        for kple in itertools.combinations(list(range(len(ids))), k)
    }

    return marginals


def block_transpose(rho):
    dim = len(rho)
    if dim == 2:
        rhot = rho.T
    else:
        h = int(dim / 2)
        rhot = np.zeros((dim, dim), dtype=complex)
        rhot[0:h, 0:h] = rho[0:h, 0:h]
        rhot[h:dim, h:dim] = rho[h:dim, h:dim]
        rhot[0:h, h:dim] = rho[h:dim, 0:h]
        rhot[h:dim, 0:h] = rho[0:h, h:dim]
    return rhot


def partial_transpose(rho, qlist, qubit, order="Standard"):

    if order == "Qiskit":
        qlist = tuple(reversed(qlist))

    dim = len(rho)
    i = qlist.index(qubit)
    k = int(dim / 2 ** i)
    rhot = np.zeros((dim, dim), dtype=complex)
    for i in range(0, dim, k):
        for j in range(0, dim, k):
            rhot[i : i + k, j : j + k] = block_transpose(rho[i : i + k, j : j + k])

    return rhot


def negativity(rho, qlist, qubit, order="Standard", atol=1e-15):
    spectrum = np.linalg.eigh(partial_transpose(rho, qlist, qubit, order=order))[0]
    return 2 * np.abs(spectrum[spectrum < atol].sum())


def negativity_list(rho, qlist, qubits, order="Standard", atol=1e-15):
    pt = rho
    for qubit in qubits:
        pt = partial_transpose(pt, qlist, qubit, order=order)
    spectrum = np.linalg.eigh(pt)[0]
    return 2 * np.abs(spectrum[spectrum < atol].sum())


def p_value(value, distribution):
    return len(np.sort(distribution)[np.sort(distribution) >= value]) / len(
        distribution
    )
