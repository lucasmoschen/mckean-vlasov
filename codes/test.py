import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

def _project_Fourier_basis_FFT_original(func, d, L, min_fourier_samples=200, project_on=None):
    """
    Original function provided by the user.
    """
    if project_on is None:
        project_on = L

    c = np.zeros(2*project_on+1, dtype=np.complex128)
    samples = 2**int(np.ceil(np.log2(max(min_fourier_samples, L + 2, project_on + 2))))
    x = (np.linspace(0, samples, samples, endpoint=False) + 0.5)*d/samples
    f = func(x)
    c_fft = (np.sqrt(d)/samples)*fft(f)*np.exp(-np.pi*1j*np.linspace(0, samples, samples, endpoint=False)/samples)
    c[project_on:] = c_fft[0: project_on+1]
    c[0:project_on] = np.conjugate(c_fft[1:project_on+1][::-1])
    return c

def _project_Fourier_basis_FFT_improved(func, d, L, min_fourier_samples=200, project_on=None):
    """
    Improved function using fftshift and proper indexing.
    """
    if project_on is None:
        project_on = L

    samples = 2**int(np.ceil(np.log2(max(min_fourier_samples, 2 * project_on + 1))))
    x = (np.arange(samples) + 0.5) * d / samples
    f = func(x)
    c_fft = fftshift(fft(f))
    k_indices = np.arange(-samples//2, samples//2)
    c_fft *= np.exp(-1j * np.pi * k_indices / samples) * np.sqrt(d) / samples
    start_idx = samples//2 - project_on
    end_idx = samples//2 + project_on + 1
    c = c_fft[start_idx:end_idx]
    return c

# Define the domain length
d = 2 * np.pi  # For example

# Define the truncation parameter
L = 50

# Define min_fourier_samples
min_fourier_samples = 256  # Choose a reasonable number

# Define a test function
def test_func(x):
    return (x - np.pi)**4 + np.exp(np.abs(x - np.pi))

# Project using the original function
coefficients_original = _project_Fourier_basis_FFT_original(
    func=test_func,
    d=d,
    L=L,
    min_fourier_samples=min_fourier_samples
)

# Project using the improved function
coefficients_improved = _project_Fourier_basis_FFT_improved(
    func=test_func,
    d=d,
    L=L,
    min_fourier_samples=min_fourier_samples
)

# Compare the coefficients
print("Original coefficients:")
print(coefficients_original)

print("\nImproved coefficients:")
print(coefficients_improved)

# Compute the difference between the coefficients
difference = coefficients_original - coefficients_improved

print("\nDifference between coefficients:")
print(difference)

# Check if the differences are within a tolerance
tolerance = 1e-12
if np.allclose(coefficients_original, coefficients_improved, atol=tolerance):
    print("\nThe coefficients are identical within the tolerance.")
else:
    print("\nThe coefficients differ by more than the tolerance.")

# Optionally, plot the coefficients
k_vals = np.arange(-L, L+1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(k_vals, np.abs(coefficients_original), linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('Original Coefficients')
plt.xlabel('k')
plt.ylabel('|c_k|')

plt.subplot(1, 2, 2)
plt.stem(k_vals, np.abs(coefficients_improved), linefmt='r-', markerfmt='ro', basefmt=' ')
plt.title('Improved Coefficients')
plt.xlabel('k')
plt.ylabel('|c_k|')

plt.tight_layout()
plt.show()

# Plot the difference
plt.figure()
plt.stem(k_vals, np.abs(difference), linefmt='g-', markerfmt='go', basefmt=' ')
plt.title('Absolute Difference between Coefficients')
plt.xlabel('k')
plt.ylabel('|Difference|')
plt.show()

