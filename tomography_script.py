import pandas as pd
import numpy as np
from qiskit import execute, Aer
import tomography.workinglib as wl
import tomography.likelihood_maximisation as lm
import networkx as nx
import tomography.hilbert_graph_tools as ht
from povm.povm_operator import POVMOperator
import qiskit.quantum_info as qi
import itertools as it

from pandarallel import pandarallel


def generate_all_povm_effects(povms, ids, k=2, order="Standard"):

    obs = {}
    for kple in it.combinations(list(range(len(ids))), k):
        obs[kple] = {}
        for ijk in wl.create_labels_list(4, k):
            i = 0
            effect = povms[kple[k - i - 1]][int(ijk[0])]
            for e in ijk[1:]:
                i += 1
                if order == "Standard":
                    effect = np.kron(povms[kple[k - i - 1]][int(e)], effect)
                else:
                    effect = np.kron(effect, povms[kple[k - i - 1]][int(e)])
            obs[kple][ijk] = effect
    return obs


def merge_results(result, Nspins, k=2):
    marginals = {}
    observables = {}
    tot_shots = sum([res["shots_per_circuit"] for res in result])
    for i, res in enumerate(result):

        # Compute all marginal distributions and k-qubit effects
        marginals_i = wl.compute_all_simplified_marginals(
            res["counts"], range(Nspins), k=k
        )
        povms_i = POVMOperator(
            [(1, qi.Pauli(label="I" * Nspins))],
            basis=None,
            atol=1e-12,
            name=None,
            povm_params=res["povm_params"],
        ).povms
        # print(res['povm_params'])
        effects_i = generate_all_povm_effects(
            povms_i, range(Nspins), order="Qiskit", k=k
        )

        for pair in marginals_i:
            if pair not in marginals:
                marginals[pair] = {}
                observables[pair] = {}
            for outcome in marginals_i[pair]:
                marginals[pair][str(i) + "_" + outcome] = marginals_i[pair][outcome]
                observables[pair][str(i) + "_" + outcome] = (
                    effects_i[pair][outcome] * res["shots_per_circuit"] / tot_shots
                )

    return marginals, observables


def concurrence_network_grad(simulation, exact, k=2):
    Nspins = int(np.log2(exact.get_statevector().shape))
    # Compute all marginal distributions and k-qubit effects
    marginals, observables = merge_results(simulation, Nspins, k=k)

    # Infer all pairwise states
    inferred_states = {}
    # concurrences = {}
    fidelities = {}
    for kple in marginals:
        inferred_states[kple] = qi.DensityMatrix(
            lm.infer_state(marginals, kple, observables[kple], tol=1e-6, maxiter=100)
        )
        if inferred_states[kple].is_valid():
            # concurrences[kple] = qi.concurrence(inferred_states[kple])
            exact_state = qi.partial_trace(
                exact.get_statevector(), [q for q in range(Nspins) if q not in kple]
            )
            fidelities[kple] = qi.state_fidelity(inferred_states[kple], exact_state)
        else:
            print("Warning: invalid state for", kple)

    # print(fidelities)
    return np.mean(list(fidelities.values()))


def kwise_fidelity(x, k=2):
    try:
        data = concurrence_network_grad(
            x.to_dict(orient="records"), x.iloc[0]["exact_state"], k=k
        )
        print("Done with", x["qubits"].iloc[0], x["method"].iloc[0], x["id"].iloc[0])
    except Exception as err:
        print(f'Problem with {x["id"].iloc[0]}', err)
        print(x)
        data = None
    return data


if __name__ == "__main__":
    pandarallel.initialize(nb_workers=10)

    hamiltonians_df = pd.DataFrame(pd.read_pickle("hamiltonians.pickle"))

    df = pd.read_json("data/counts_data.txt", lines=True)
    df = pd.merge(
        df,
        hamiltonians_df[["molecule", "mapping", "name", "vqe_circuit"]],
        left_on="name",
        right_on="name",
        how="left",
        sort=False,
    )

    df["exact_state"] = df["vqe_circuit"].apply(
        lambda x: execute(x, Aer.get_backend("statevector_simulator")).result()
    )

    sample_df = df
    avg_fidelities_k = {}

    for k in range(2, 6):
        print(k)
        avg_fidelities_k[k] = sample_df.groupby(
            ["qubits", "method", "id"]
        ).parallel_apply(lambda x: kwise_fidelity(x, k))

    dfs = []
    for k in avg_fidelities_k:
        fid_df = pd.DataFrame(avg_fidelities_k[k], columns=["avg_fidelity"])
        fid_df["k"] = k
        dfs.append(fid_df)

    fid_df = pd.concat(dfs)

    fid_df["infidelity"] = 1 - fid_df["avg_fidelity"]

    filename = "data/fidelities_tomo.json"
    fid_df.reset_index().to_json(filename, orient="records")
    print("Written to", filename)
