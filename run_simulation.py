""" Compare POVM vs Pauli with chemistry Hamiltonians """
import argparse
import os
import pickle

import logging

import shortuuid

import numpy as np

from povm.estimator import (
    POVMEstimator,
    PauliEstimator,
    GroupedPauliEstimator,
    GradPOVMEstimator,
    GooglePOVMEstimator,
    GoogleGradPOVMEstimator,
)

# import pandas as pd
# from time import time
import json

import subprocess

from filelock import FileLock

from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("qiskit").setLevel(logging.WARNING)


def get_git_revision_short_hash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nqubits", "--nq", type=int)
    parser.add_argument("--file", type=str, default="hamiltonians.pickle")
    parser.add_argument("--hamiltonian", type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--shots", type=int, nargs="+")
    parser.add_argument("--method", type=str)
    parser.add_argument("--outfile", type=str)  # argparse.FileType('a'))
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--counts", action="store_true")
    args = parser.parse_args()

    commit = get_git_revision_short_hash()

    methods_dict = {
        "SIC-POVM": POVMEstimator,
        "Pauli": PauliEstimator,
        "Grouped_Pauli": GroupedPauliEstimator,
        "Grad-POVM": GradPOVMEstimator,
        "Google-POVM": GooglePOVMEstimator,
        "Grad-Google-POVM": GoogleGradPOVMEstimator,
    }

    data = pickle.load(open(args.file, "rb"))

    h = data[args.hamiltonian]

    qubitOp = h["operator"]

    if args.exact:
        qc = h["exact_circuit"]
        exact = h["exact_value"]
    else:
        qc = h["vqe_circuit"]
        exact = h["vqe_value"]

    estimator = methods_dict[args.method](qc, qubitOp, exact, return_counts=args.counts)

    for i in range(args.samples):
        for shots in args.shots:

            estimator_id = shortuuid.uuid()

            logging.info(
                "Starting %s, %s, %d shots (%s). %s",
                h["name"],
                args.method,
                shots,
                estimator_id,
                "exact" if args.exact else "",
            )

            try:
                for result in estimator.estimate(shots):
                    result["name"] = h["name"]
                    result["commit"] = commit
                    result["exact_ground_state"] = args.exact
                    result["id"] = estimator_id
                    result["timestamp"] = datetime.timestamp(datetime.now())

                    # filename = f'{h["Type"]}_{h["Qubits"]}_{args.method}_{result["Shots"]}_{estimator_id}.json'
                    logging.info(
                        "Done with %d, %s, %s, %d shots (%s).",
                        result["qubits"],
                        result["name"],
                        result["method"],
                        result["shots"],
                        estimator_id,
                    )

                    # with FileLock(args.outfile.name + ".lock"):
                    with FileLock(args.outfile + ".lock"):
                        with open(args.outfile, "a") as f:
                            # args.outfile.write(json.dumps(result) + '\n')
                            f.write(json.dumps(result) + "\n")

            except Exception as ex:
                logging.error(
                    "Something happened to %s with with %s: %s",
                    args.method,
                    h["name"],
                    ex,
                    exc_info=ex,
                )

    logging.info("Done, exiting.")
