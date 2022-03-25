#!/usr/bin/env python3
"""Create a plot for vegas_accuracies.py output"""
import sys

sys.path.append("..")

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_measurements(measurement_file_path, mode):
    # Do not show top and right border in plots
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    # Bold titles
    # ~ plt.rcParams["axes.titleweight"] = "bold"

    df = pd.read_csv(measurement_file_path)
    for integrand_name, df in df.groupby(["integrand"]):
        output_folder = Path("./generated_plots")
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = str(output_folder / f"vegas_{mode}_{integrand_name}.pdf")
        print(f"Generating the plot {output_file}.")

        fig, ax = plt.subplots()
        ax.set_title(f"VEGAS implementation comparison, integrand {integrand_name}")

        for (integrand_name, dim, implementation), df in df.groupby(
            ["integrand", "dim", "implementation"]
        ):
            data = df.to_dict(orient="list")
            x = data["num_evals"]
            yid = {"accuracy": "error_rel", "time": "time"}[mode]
            y = data[yid]
            colour = {
                "MonteCarlo": "#b3a59d",
                "torchquad": "#EC0E0F",
                "gplepage": "#1F77B4",
                "VegasFlow": "#006500",
            }.get(implementation, None)
            marker = {
                "MonteCarlo": "x",
                "torchquad": "+",
                "gplepage": "3",
                "VegasFlow": ".",
            }.get(implementation, None)
            ax.scatter(
                x,
                y,
                c=colour,
                s=30,
                alpha=0.7,
                marker=marker,
                label=f"{dim}D, {implementation}",
            )

        ylabel = {"accuracy": "Relative error", "time": "Time in s"}[mode]
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Number of integrand evaluations")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output_file, bbox_inches="tight")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-file",
        help="CSV file where the measurements are saved",
        default="tmp_vegas_measurements.csv",
    )
    parser.add_argument(
        "--mode",
        help="Configure what to plot",
        choices=["accuracy", "time"],
        default="accuracy",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    plot_measurements(args.csv_file, args.mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
