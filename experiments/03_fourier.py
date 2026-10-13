
import matplotlib.pyplot as plt
import numpy as np


# --- Configuration ---

SAMPLE_RATE = 2000
DURATION = 1.0
TOP_N = 3


# --- Signal generation ---

def generate_signal(t):
    return (
        np.sin(2 * np.pi * 100 * t)
        + 0.6 * np.sin(2 * np.pi * 250 * t)
        + 0.3 * np.sin(2 * np.pi * 440.5 * t)
    )


# --- Frequency analysis ---

def compute_spectrum(signal, sample_rate):
    spectrum = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / sample_rate,
    )

    amplitudes = 2 * np.abs(spectrum) / len(signal)

    # DC and Nyquist bins are not doubled.
    amplitudes[0] /= 2
    if len(signal) % 2 == 0:
        amplitudes[-1] /= 2

    return frequencies, amplitudes


def dominant_frequencies(frequencies, amplitudes, top_n=3):
    components = zip(frequencies, amplitudes)

    return sorted(
        components,
        key=lambda component: component[1],
        reverse=True,
    )[:top_n]


# --- Visualization ---

def plot_spectrum(frequencies, amplitudes):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(frequencies, amplitudes)

    ax.set_xlim(0, 600)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Frequency Spectrum")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# --- Entry point ---

def main():
    t = np.arange(
        int(SAMPLE_RATE * DURATION)
    ) / SAMPLE_RATE

    signal = generate_signal(t)

    frequencies, amplitudes = compute_spectrum(
        signal, SAMPLE_RATE
    )

    components = dominant_frequencies(
        frequencies, amplitudes, TOP_N
    )

    print(f"Duration: {DURATION} s")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"FFT bin spacing: {1 / DURATION:.2f} Hz")

    print("\nDominant frequency bins:")
    for frequency, amplitude in components:
        print(f"  {frequency:7.2f} Hz | amplitude: {amplitude:.4f}")

    plot_spectrum(frequencies, amplitudes)
    plt.show()


if __name__ == "__main__":
    main()
