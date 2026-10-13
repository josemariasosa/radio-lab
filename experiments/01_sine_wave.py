import numpy as np
import matplotlib.pyplot as plt
import wave

# Signal parameters
# amplitude = 0.8
amplitude = 0.2
frequency = 440.0
# frequency = 880.0
# phase = 0.0
phase = 3.141592/2

# Sampling parameters
# sample_rate = 44_100
sample_rate = 100
# sample_rate = 88_000
duration = 1.0

# Discrete time axis
t = np.arange(int(sample_rate * duration)) / sample_rate

# Generate the wave
signal = amplitude * np.sin(
    2 * np.pi * frequency * t + phase
)

# Plot only the first 10 milliseconds
samples_to_plot = int(sample_rate * 0.010)

plt.plot(
    t[:samples_to_plot],
    signal[:samples_to_plot],
)

plt.title("Our first sine wave — 440 Hz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Convert normalized audio to 16-bit PCM
pcm = (signal * 32767).astype(np.int16)

with wave.open("tone_440hz_02apl.wav", "wb") as wav:
# with wave.open("tone_880hz.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(pcm.tobytes())