import numpy as np
from scipy.fft import fft, fftshift

class FourierUtils:
    """
    A class for working with Fourier series.
    """
    def __init__(self, L, d, min_fourier_samples=200):
        self.L = L
        self.d = d
        self.min_fourier_samples = min_fourier_samples
        self.k_vals = np.arange(-L, L + 1)
        self.N = len(self.k_vals)
        self.phi_funcs = [self._phi_k(k) for k in self.k_vals]

    def _phi_k(self, k):
        """Return the k-th orthonormal Fourier basis function on [0, d]."""
        return lambda x: np.exp(2 * np.pi * 1j * k * x / self.d) / np.sqrt(self.d)

    def integrate(self, f, n_points=1000):
        """Numerical integration over [0, d] using the trapezoidal rule."""
        x_vals = np.linspace(0, self.d, n_points)
        y_vals = f(x_vals)
        return np.trapezoid(y_vals, x_vals)

    def project_function(self, func, project_on=None, d=None):
        """
        Project a function onto the Fourier basis using FFT.
        Uses fftshift for proper indexing.
        """
        if project_on is None:
            project_on = self.L
        if d is None:
            d = self.d

        samples = 2 ** int(np.ceil(np.log2(max(self.min_fourier_samples, 2 * project_on + 1))))
        x = (np.arange(samples) + 0.5) * d / samples
        f = func(x)
        c_fft = fftshift(fft(f))
        k_indices = np.arange(-samples // 2, samples // 2)
        c_fft *= np.exp(-1j * np.pi * k_indices / samples) * np.sqrt(d) / samples
        start_idx = samples // 2 - project_on
        end_idx = samples // 2 + project_on + 1
        return c_fft[start_idx:end_idx]

    def reconstruction(self, a, x):
        """
        Reconstruct a function from coefficients a and points x.
        """
        return np.real(np.sum([a[k]*self.phi_funcs[k](x) for k in range(self.N)], axis=0))

    def conjugate_wrapper(self, a, c=0.0):
        """
        Convert the real-valued vector into a conjugate-symmetric complex form.
        """
        return np.hstack([a[:self.L][::-1] - 1j * a[self.L:][::-1], c, a[:self.L] + 1j * a[self.L:]])

    def real_wrapper(self, a, c=0.0):
        """
        Convert the conjugate-symmetric complex vector into a real vector.
        """
        return np.hstack([a[self.L + 1:].real, a[self.L + 1:].imag])
    
    def conjugate_wrapper_matrix(self, a, c=0.0):
        """
        Convert a matrix whose columns are in the real representation (each of length 2*self.L)
        into a matrix of complex vectors (each of length 2*self.L+1).
        """
        L = self.L
        output = np.empty((2 * L + 1, a.shape[1]), dtype=np.complex128)
        output[:L, :] = a[:L, :][::-1] - 1j * a[L:, :][::-1]
        output[L, :] = c
        output[L+1:, :] = a[:L, :] + 1j * a[L:, :]
        return output
