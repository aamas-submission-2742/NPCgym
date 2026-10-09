"""Plot the eight bundled Merchant baseline runs; requires the plots extra."""

from pathlib import Path

from npc_gym.evaluation import plot_learning_curve


def main() -> None:
    """Plot the bundled CSVs when called or executed as a script."""
    results = Path("experiments/results/merchant/merchant-tabular-unconstrained-v2")
    curves = [results / f"seed-{seed}/learning_curve.csv" for seed in range(8)]
    output = Path("experiments/output/merchant-learning-curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(
        curves,
        output,
        counts=[("merchant/delivery-v0", "count"), ("merchant/env-friendly-v0", "count")],
    )
    print(output)


if __name__ == "__main__":
    main()
