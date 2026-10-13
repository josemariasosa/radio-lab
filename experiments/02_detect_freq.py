import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# --- Configuration ---

SAMPLE_RATE = 2000
FREQUENCY = 440
DURATION = 0.05

NOISE_SCALES = (0.15, 0.3, 0.5, 1.0)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
N_SEEDS = 1000


# --- Signal generation ---

def generate_signal(t, frequency, noise_scale, seed):
    rng = np.random.default_rng(seed)

    wave = np.sin(2 * np.pi * frequency * t)
    noise = rng.normal(0, noise_scale, size=len(t))

    return wave + noise


# --- Frequency estimation ---

def detect_sign_change(a, b):
    return (a >= 0 > b) or (a < 0 <= b)


def estimate_zero_crossing(t1, t2, y1, y2):
    return t1 - y1 * (t2 - t1) / (y2 - y1)


def frequency_from_crossings(crossings):
    if len(crossings) < 2:
        return float("nan")

    mean_interval = np.mean(np.diff(crossings))
    return 1 / (2 * mean_interval)


def naive_zero_crossing_estimate(signal, t):
    crossings = []

    for i in range(1, len(signal)):
        if detect_sign_change(signal[i - 1], signal[i]):
            crossing = estimate_zero_crossing(
                t[i - 1], t[i],
                signal[i - 1], signal[i],
            )
            crossings.append(crossing)

    return frequency_from_crossings(crossings)


def threshold_zero_crossing_estimate(signal, t, threshold):
    crossings = []
    last_accepted = None

    for time, amplitude in zip(t, signal):
        if abs(amplitude) < threshold:
            continue

        if last_accepted is None:
            last_accepted = (time, amplitude)
            continue

        prev_time, prev_amplitude = last_accepted

        if detect_sign_change(prev_amplitude, amplitude):
            crossing = estimate_zero_crossing(
                prev_time, time,
                prev_amplitude, amplitude,
            )
            crossings.append(crossing)
            last_accepted = (time, amplitude)

    return frequency_from_crossings(crossings)


# --- Benchmark & analysis ---

def run_benchmark(t, frequency, scales, thresholds, n_seeds):
    results = []

    for scale in scales:
        for seed in range(n_seeds):
            signal = generate_signal(t, frequency, scale, seed)

            naive = naive_zero_crossing_estimate(signal, t)

            for threshold in thresholds:
                schmitt = threshold_zero_crossing_estimate(
                    signal, t, threshold
                )

                results.append({
                    "scale": scale,
                    "threshold": threshold,
                    "seed": seed,
                    "naive_estimate": naive,
                    "schmitt_estimate": schmitt,
                    "naive_error": abs(naive - frequency),
                    "schmitt_error": abs(schmitt - frequency),
                })

    return pl.DataFrame(results)


def summarize_results(df):
    return (
        df.group_by("scale", "threshold")
        .agg(
            pl.len().alias("samples"),
            pl.col("naive_error").mean().alias("naive_mae"),
            pl.col("schmitt_error").mean().alias("schmitt_mae"),
            pl.col("naive_error").max().alias("naive_worst"),
            pl.col("schmitt_error").max().alias("schmitt_worst"),
        )
        .with_columns(
            (
                pl.col("naive_mae") - pl.col("schmitt_mae")
            ).alias("mae_improvement")
        )
        .sort(["scale", "threshold"])
    )


# --- Visualization ---

def heatmap_matrix(summary, metric):
    pivot = (
        summary
        .pivot(
            on="threshold",
            index="scale",
            values=metric,
        )
        .sort("scale")
    )

    thresholds = sorted(
        (c for c in pivot.columns if c != "scale"),
        key=float,
    )

    return (
        pivot["scale"].to_list(),
        thresholds,
        pivot.select(thresholds).to_numpy(),
    )


def plot_heatmap(summary, metric, title, cmap, centered=False):
    scales, thresholds, matrix = heatmap_matrix(summary, metric)

    fig, ax = plt.subplots(figsize=(9, 5))

    options = {}
    if centered:
        limit = np.nanmax(np.abs(matrix))
        if limit > 0:
            options = {"vmin": -limit, "vmax": limit}

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        **options,
    )

    ax.set_xticks(range(len(thresholds)), thresholds)
    ax.set_yticks(range(len(scales)), scales)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Noise scale")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = f"{value:+.1f}" if centered else f"{value:.1f}"

            ax.text(
                j, i, label,
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Error (Hz)")
    fig.tight_layout()
    return fig


def plot_error_heatmap(summary):
    return plot_heatmap(
        summary,
        metric="schmitt_mae",
        title="Threshold detector - MAE (Hz)",
        cmap="YlOrRd",
    )


def plot_improvement_heatmap(summary):
    return plot_heatmap(
        summary,
        metric="mae_improvement",
        title="Threshold improvement over Naive (Hz)",
        cmap="RdBu",
        centered=True,
    )


# --- Entry point ---

def main():
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE

    results = run_benchmark(
        t,
        frequency=FREQUENCY,
        scales=NOISE_SCALES,
        thresholds=THRESHOLDS,
        n_seeds=N_SEEDS,
    )

    summary = summarize_results(results)

    print("\n=== Error summary (Hz) ===")
    print(summary)

    plot_error_heatmap(summary)
    plot_improvement_heatmap(summary)

    plt.show()


if __name__ == "__main__":
    main()
