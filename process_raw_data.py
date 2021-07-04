import pandas as pd
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON Line raw data files to a feather dataset"
    )
    parser.add_argument(
        "src_path",
        metavar="path",
        type=str,
        help="Path to files to be processed; enclose in quotes, accepts * as wildcard for directories or filenames",
    )
    parser.add_argument("--output", "-o", type=str, help="Output file", required=True)
    parser.add_argument(
        "--hamiltonians",
        type=str,
        default="hamiltonians.pickle",
        help="File with the Hamiltonians",
    )
    args = parser.parse_args()

    files = glob.glob(args.src_path)

    print(len(files), "files to be processed.")

    dfs = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(dfs)

    df_h = pd.DataFrame(pd.read_pickle(args.hamiltonians))

    df = pd.merge(
        df,
        df_h[
            [
                "molecule",
                "basis",
                "two_qubit_reduction",
                "mapping",
                "z2symmetry_reduction",
                "freeze_core",
                "name",
            ]
        ],
        left_on="name",
        right_on="name",
        how="left",
        sort=False,
    )

    print(df.info())
    df.to_feather(args.output)
