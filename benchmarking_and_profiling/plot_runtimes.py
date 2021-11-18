#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import argparse
from pathlib import Path


BACKEND_NAMES = {
    "jax": "JAX",
    "numpy": "Numpy",
    "torch": "PyTorch",
    "tensorflow": "Tensorflow",
}
BACKEND_COLOURS = {
    "numpy": "#b3a59d",
    "torch": "#E62969",
    "jax": "#4B6FFE",
    "tensorflow": "#007B00",
}
# Alternative colours to use for plots in the background
BACKEND_COLOURS_BRIGHT = {
    "numpy": "#FF00CA",
    "torch": "#ffced6",
    "jax": "#c3d4ff",
    "tensorflow": "#c7f4c3",
}
BACKEND_COLOURS_GREY = {
    "numpy": "#FF00CA",
    "torch": "#A38C81",
    "jax": "#8AA0A3",
    "tensorflow": "#ABBE99",
}
COMPILED_INFO_NAMES = {
    "no_unsupported": "uncompiled (unsupported)",
    "no": "uncompiled",
    "three_steps": "parts compiled",
    "integrate": "all compiled",
    "step1": "compiled",
    "step3": "compiled",
}


def backend_legend(backend, cuda, gradient=False):
    """Get a string which shows the numerical backend and if it uses CUDA"""
    name = BACKEND_NAMES[backend]
    if cuda is None:
        return name
    ext = " (CPU)"
    if cuda and backend in ["torch", "jax", "tensorflow"]:
        ext = " (CUDA)"
    if gradient:
        assert cuda, "Not implemented"
        ext = ", gradient (CUDA)"
    return name + ext


def plot_lines_ax(
    ax,
    xs,
    ys,
    legend_elements,
    colours,
    linestyles,
    title,
    ylabel,
    xlabel,
    xscale_log=False,
    yscale_log=False,
    plotconf={},
):
    """Plot lines on a given axes"""
    if title is not None:
        ax.set_title(title)

    xlim_left = plotconf.get("xlim_left", None)
    for x, y, col, linestyle in zip(xs, ys, colours, linestyles):
        if xlim_left:
            num_invisible = x[x < xlim_left].shape[0]
            if x.shape == (num_invisible,):
                continue
            x = x[num_invisible - 1 :]
            y = y[num_invisible - 1 :]
        ax.plot(x, y, linestyle, markersize=3.5, c=col)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xscale_log:
        ax.set_xscale("log")
    xlim_right = plotconf.get("xlim_right", None)
    ylim_bottom = plotconf.get("ylim_bottom", None)
    ylim_top = plotconf.get("ylim_top", None)
    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)
    if xlim_right is not None:
        ax.set_xlim(right=xlim_right)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)
    if yscale_log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if legend_elements is not None:
        if plotconf.get("legend_outside", True):
            # Consistent location of the legend right to the plot
            ax.legend(
                handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5)
            )
        else:
            # Legend inside the Plot; requires less whitespace but may work badly
            # with certain inputs
            ax.legend(handles=legend_elements, loc="best")


def plot_lines(
    xs,
    ys,
    legend_elements,
    colours,
    linestyles,
    title,
    ylabel,
    xlabel,
    xscale_log=False,
    yscale_log=False,
    plotconf={},
    output_file=None,
):
    """Create line plots"""
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_lines_ax(
        ax,
        xs,
        ys,
        legend_elements,
        colours,
        linestyles,
        title,
        ylabel,
        xlabel,
        xscale_log,
        yscale_log,
        plotconf,
    )

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")


def plot_runtimes(df, title, output_file, dim, plotconf):
    """
    Plot measurements collected with the measure_runtimes script to compare
    different compilation options and backends
    """
    print(f"Generating plot {output_file}")
    backends = []
    compiled_infos = []
    Nss = []
    runtimess = []
    for (backend, compiled_info), df2 in df.groupby(["backend", "compiled_info"]):
        if compiled_info == "no_unsupported":
            continue
        backends.append(backend)
        compiled_infos.append(compiled_info)
        Nss.append(np.array(df2["N"]))
        runtimess.append(np.array(df2["median_runtime"]))

    # Get colours and styles for the lines
    # ~ dots_for_backend = {"numpy": "4", "torch": "3", "jax": "2", "tensorflow": "1"}
    dots_for_backend = {"numpy": ".", "torch": ".", "jax": ".", "tensorflow": "."}
    dashing_for_compilation = {
        "no": "-",
        "three_steps": "-.",
        "integrate": "--",
        "step1": "--",
        "step3": "--",
    }
    colours = [BACKEND_COLOURS[b] for b in backends]
    linestyles = []
    for compiled_info, backend in zip(compiled_infos, backends):
        linestyles.append(
            dots_for_backend[backend] + dashing_for_compilation[compiled_info]
        )

    # Get legend elements
    legend_elems = []
    for backend in ["numpy", "torch", "jax", "tensorflow"]:
        if backend not in backends:
            continue
        backend_str = backend_legend(backend, plotconf["cuda_used"])
        legend_elems.append(
            Line2D([0], [0], color=BACKEND_COLOURS[backend], lw=4, label=backend_str)
        )
    for compiled_info in ["no", "three_steps", "integrate", "step1", "step3"]:
        if compiled_info not in compiled_infos:
            continue
        compiled_info_str = COMPILED_INFO_NAMES[compiled_info]
        legend_elems.append(
            Line2D(
                [0],
                [0],
                linestyle=dashing_for_compilation[compiled_info],
                color="black",
                label=compiled_info_str,
            )
        )

    show_n_per_dim = False
    xss, xlabel = None, None
    if show_n_per_dim:
        xss = [xs ** (1.0 / dim) for xs in Nss]
        xlabel = "N per dim"
    else:
        xss = Nss
        xlabel = "Number of evaluations"

    # Create the plot
    plot_lines(
        xss,
        runtimess,
        colours=colours,
        linestyles=linestyles,
        legend_elements=legend_elems,
        title=title,
        ylabel="Median time in s",
        xlabel=xlabel,
        xscale_log=True,
        yscale_log=True,
        plotconf=plotconf,
        output_file=output_file,
    )


def plot_precision_comparison(df, title, output_file, plotconf):
    """
    Plot measurements collected with the measure_runtimes script to compare
    different precisions
    """
    print(f"Generating plot {output_file}")
    backends = []
    precisions = []
    Nss = []
    runtimess = []
    for (backend, precision_dtype), df2 in df.groupby(["backend", "precision_dtype"]):
        backends.append(backend)
        precisions.append(precision_dtype)
        Nss.append(np.array(df2["N"]))
        runtimess.append(np.array(df2["median_runtime"]))

    # Get colours and styles for the lines
    dots_for_backend = {"numpy": ".", "torch": ".", "jax": ".", "tensorflow": "."}
    dashing_for_precision = {"float16": "-", "float32": "--", "float64": ":"}
    colours = [BACKEND_COLOURS[b] for b in backends]
    linestyles = []
    for precision, backend in zip(precisions, backends):
        linestyles.append(dots_for_backend[backend] + dashing_for_precision[precision])

    # Get legend elements
    legend_elems = []
    for backend in ["numpy", "torch", "jax", "tensorflow"]:
        if backend not in backends:
            continue
        backend_str = backend_legend(backend, plotconf["cuda_used"])
        legend_elems.append(
            Line2D([0], [0], color=BACKEND_COLOURS[backend], lw=4, label=backend_str)
        )
    for precision in ["float16", "float32", "float64"]:
        if precision not in precisions:
            continue
        legend_elems.append(
            Line2D(
                [0],
                [0],
                linestyle=dashing_for_precision[precision],
                color="black",
                label=precision,
            )
        )

    # Create the plot
    plot_lines(
        Nss,
        runtimess,
        colours=colours,
        linestyles=linestyles,
        legend_elements=legend_elems,
        title=title,
        ylabel="Median time in s",
        xlabel="Number of evaluations",
        xscale_log=True,
        yscale_log=True,
        plotconf=plotconf,
        output_file=output_file,
    )


def plot_gradient_comparison(
    df_nograd, df_grad, title, output_file, plotconf, four_plots
):
    """
    Plot measurements collected with the measure_runtimes script to compare
    calculation with and without gradient
    """
    print(f"Generating plot {output_file}")
    dashing_for_compilation = {
        "no": "-",
        "three_steps": "-.",
        "integrate": "--",
        "step1": "--",
        "step3": "--",
    }
    backends = []
    colours = []
    linestyles = []
    compiled_infos = []
    Nss = []
    runtimess = []
    is_grad = []
    colours_alt = BACKEND_COLOURS_GREY if four_plots else BACKEND_COLOURS_BRIGHT
    for (backend, compiled_info), df2 in df_nograd.groupby(
        ["backend", "compiled_info"]
    ):
        if backend == "numpy":
            continue
        backends.append(backend)
        colours.append(colours_alt[backend])
        compiled_infos.append(compiled_info)
        linestyles.append("H" + dashing_for_compilation[compiled_info])
        Nss.append(np.array(df2["N"]))
        runtimess.append(np.array(df2["median_runtime"]))
        is_grad.append(False)

    for (backend, compiled_info), df2 in df_grad.groupby(["backend", "compiled_info"]):
        backends.append(backend)
        colours.append(BACKEND_COLOURS[backend])
        compiled_infos.append(compiled_info)
        linestyles.append("." + dashing_for_compilation[compiled_info])
        Nss.append(np.array(df2["N"]))
        runtimess.append(np.array(df2["median_runtime"]))
        is_grad.append(True)

    # Get legend elements
    legend_elems = []
    for gradient in (True, False):
        for backend in ["torch", "jax", "tensorflow"]:
            if backend not in backends:
                continue
            backend_str = backend_legend(
                backend, plotconf["cuda_used"], gradient=gradient
            )
            colour = BACKEND_COLOURS[backend] if gradient else colours_alt[backend]
            legend_elems.append(Line2D([0], [0], color=colour, lw=4, label=backend_str))
    for compiled_info in ["no", "three_steps", "integrate", "step1", "step3"]:
        if compiled_info not in compiled_infos:
            continue
        compiled_info_str = COMPILED_INFO_NAMES[compiled_info]
        legend_elems.append(
            Line2D(
                [0],
                [0],
                linestyle=dashing_for_compilation[compiled_info],
                color="black",
                label=compiled_info_str,
            )
        )

    if not four_plots:
        plot_lines(
            Nss,
            runtimess,
            colours=colours,
            linestyles=linestyles,
            legend_elements=legend_elems,
            title=title,
            ylabel="Median time in s",
            xlabel="Number of evaluations",
            xscale_log=True,
            yscale_log=True,
            plotconf=plotconf,
            output_file=output_file,
        )
    else:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        # ~ fig, axs = plt.subplots(2, 2)
        common_kargs = {
            "legend_elements": None,
            "title": None,
            "ylabel": "Median time in s",
            "xlabel": "Number of evaluations",
            "xscale_log": True,
            "yscale_log": True,
            "plotconf": plotconf,
        }

        xss, yss, colss, linestyless = (
            {"grad": [], "torch": [], "jax": [], "tensorflow": []} for _ in range(4)
        )
        for x, y, col, ls, g, backend in zip(
            Nss, runtimess, colours, linestyles, is_grad, backends
        ):
            if g:
                xss["grad"].append(x)
                yss["grad"].append(y)
                colss["grad"].append(col)
                linestyless["grad"].append(ls)
            xss[backend].append(x)
            yss[backend].append(y)
            colss[backend].append(col)
            linestyless[backend].append(ls)
        ax_order = {
            "grad": axs[0][0],
            "torch": axs[0][1],
            "jax": axs[1][0],
            "tensorflow": axs[1][1],
        }
        for k, ax in ax_order.items():
            plot_lines_ax(
                ax,
                xss[k],
                yss[k],
                colours=colss[k],
                linestyles=linestyless[k],
                **common_kargs,
            )

        fig.suptitle(title)
        fig.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.0, 0.5))

        plt.tight_layout()

        if output_file is not None:
            fig.savefig(output_file, bbox_inches="tight")


def make_plots(mode, measurements_csv_path, plotconf):
    df = pd.read_csv(measurements_csv_path)
    output_folder = Path("./generated_plots")
    output_folder.mkdir(parents=True, exist_ok=True)
    all_integrator_names = ["Trapezoid", "Simpson", "Boole", "MonteCarlo"]
    if mode == "compilations":
        for (integrator_name, dim, integrand, precision_dtype), df in df.groupby(
            ["integrator_name", "dim", "integrand", "precision_dtype"]
        ):
            if integrator_name not in all_integrator_names:
                continue
            title = f"Compilation comparison, {integrator_name}, {dim}D {integrand}, {precision_dtype}"
            filename = f"compilations_{integrator_name}_{dim}_{integrand}_{precision_dtype}.pdf"
            plot_runtimes(df, title, str(output_folder / filename), dim, plotconf)
    elif mode == "integrate_parts":
        for (integrator_name, dim, integrand, precision_dtype), df in df.groupby(
            ["integrator_name", "dim", "integrand", "precision_dtype"]
        ):
            if integrator_name in all_integrator_names:
                continue
            # Hide the integrand in step1 and step3
            title = (
                f"Compilation comparison, {integrator_name}, {dim}D, {precision_dtype}"
            )
            filename = (
                f"parts_{integrator_name}_{dim}_{integrand}_{precision_dtype}.pdf"
            )
            plot_runtimes(df, title, str(output_folder / filename), dim, plotconf)
    elif mode == "precisions":
        for (integrator_name, dim, integrand, compiled_info), df in df.groupby(
            ["integrator_name", "dim", "integrand", "compiled_info"]
        ):
            if "_step" in integrator_name:
                # Hide the integrand in step1 and step3
                title = f"Precision comparison, {integrator_name}, {dim}D, {COMPILED_INFO_NAMES[compiled_info]}"
            else:
                title = f"Precision comparison, {integrator_name}, {dim}D {integrand}, {COMPILED_INFO_NAMES[compiled_info]}"
            filename = (
                f"precisions_{integrator_name}_{dim}_{integrand}_{compiled_info}.pdf"
            )
            plot_precision_comparison(
                df, title, str(output_folder / filename), plotconf
            )
    if mode == "gradients_compare":
        for (integrator_name, dim, precision_dtype), df in df.groupby(
            ["integrator_name", "dim", "precision_dtype"]
        ):
            if integrator_name not in all_integrator_names:
                continue
            df_nograd = None
            df_grad = None
            integrand_name = None
            for integrand, df in df.groupby(["integrand"]):
                if integrand.endswith("(gradient)"):
                    if df_grad is not None:
                        raise RuntimeError("multiple integrands found in csv file")
                    df_grad = df
                else:
                    if df_nograd is not None:
                        raise RuntimeError("multiple integrands found in csv file")
                    integrand_name = integrand
                    df_nograd = df
            title = f"Gradient calculation comparison, {integrator_name}, {dim}D {integrand_name}, {precision_dtype}"
            filename = f"gradient_comparison_{integrator_name}_{dim}_{integrand}_{precision_dtype}.pdf"
            plot_gradient_comparison(
                df_nograd, df_grad, title, str(output_folder / filename), plotconf, True
            )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file",
        help="CSV file path where the measurements are saved",
    )
    parser.add_argument(
        "--mode",
        help="Type of comparison plots which should be generated",
        choices=["compilations", "precisions", "integrate_parts", "gradients_compare"],
        default="compilations",
    )
    parser.add_argument(
        "--cuda-used",
        help="Specify if CUDA was used for JAX, Torch and Tensorflow",
        choices=[True, False, None],
        type=lambda a: {"True": True, "False": False, None: None}[a],
        default=None,
    )
    parser.add_argument(
        "--xlim-left",
        help="Start of the x axis",
        default="auto",
        type=lambda x: None if x == "auto" else float(x),
    )
    parser.add_argument(
        "--xlim-right",
        help="End of the x axis",
        default="auto",
        type=lambda x: None if x == "auto" else float(x),
    )
    parser.add_argument(
        "--ylim-bottom",
        help="Start of the y axis",
        default="auto",
        type=lambda x: None if x == "auto" else float(x),
    )
    parser.add_argument(
        "--ylim-top",
        help="End of the y axis",
        default="auto",
        type=lambda x: None if x == "auto" else float(x),
    )
    parser.add_argument(
        "--legend-inside",
        help="Show the legend in the plot instead of right to it",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    make_plots(
        args.mode,
        args.data_file,
        {
            "xlim_left": args.xlim_left,
            "xlim_right": args.xlim_right,
            "ylim_bottom": args.ylim_bottom,
            "ylim_top": args.ylim_top,
            "cuda_used": args.cuda_used,
            "legend_outside": not args.legend_inside,
        },
    )


main()
