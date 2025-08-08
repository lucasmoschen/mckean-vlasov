import numpy as np
from scipy.fft import fft, fftshift
from numpy.fft import fft2, ifft2, fftshift, ifftshift

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
        Project a function onto the Fourier basis using FFT,
        using the trapezoidal rule for integration.
        """
        if project_on is None:
            project_on = self.L
        if d is None:
            d = self.d

        samples = 2 ** int(np.ceil(np.log2(max(self.min_fourier_samples, 2 * project_on + 1))))
        # Trapezoidal rule: sample on the grid 0, d/samples, ..., d*(samples-1)/samples.
        x = np.arange(samples) * d / samples
        f = func(x)
        c_fft = fftshift(fft(f))
        # Remove the extra phase factor (only needed for midpoint sampling)
        c_fft *= np.sqrt(d) / samples
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

class FourierUtils2D:
    """
    2D Fourier utilities with product basis on [0,d] x [0,d].
    Modes are kx in [-Lx..Lx], ky in [-Ly..Ly], DC mode is 1/d.
    Real/complex packing uses the standard 2D half-lattice:
      S = {(kx,ky): kx>0 or (kx==0 and ky>0)}; DC excluded.
    """
    def __init__(self, L_tuple, d, min_fourier_samples=200):
        self.Lx, self.Ly = L_tuple
        self.d = d
        self.min_fourier_samples = min_fourier_samples

        self.kx_vals = np.arange(-self.Lx, self.Lx + 1)
        self.ky_vals = np.arange(-self.Ly, self.Ly + 1)
        self.Nx = len(self.kx_vals)
        self.Ny = len(self.ky_vals)
        self.N = self.Nx * self.Ny

        KX, KY = np.meshgrid(self.kx_vals, self.ky_vals, indexing='ij')
        self.KX = KX
        self.KY = KY
        self.KX_flat = KX.ravel()
        self.KY_flat = KY.ravel()

        # Indexing helpers
        self.zero_flat = self._flat_index(0, 0)
        self.pos_mask = (self.KX > 0) | ((self.KX == 0) & (self.KY > 0))
        self.pos_indices = np.flatnonzero(self.pos_mask.ravel())
        self.npos = self.pos_indices.size

        # Map positive indices to their conjugate negatives
        neg_map = {}
        for idx in self.pos_indices:
            kx = self.KX_flat[idx]; ky = self.KY_flat[idx]
            neg_map[idx] = self._flat_index(-kx, -ky)
        self.neg_index_of_pos = np.array([neg_map[idx] for idx in self.pos_indices], dtype=int)

        # Basis functions list (kept for API parity; used sparsely)
        self.phi_funcs = [self._phi_k(kx, ky) for kx, ky in zip(self.KX_flat, self.KY_flat)]

    def _flat_index(self, kx, ky):
        ix = int(kx + self.Lx)
        iy = int(ky + self.Ly)
        return ix * self.Ny + iy

    def _phi_k(self, kx, ky):
        """Return φ_{(kx,ky)}(x,y) = exp(2πi (kx x + ky y)/d) / d."""
        def phi(X, Y):
            return np.exp(2j * np.pi * (kx * X + ky * Y) / self.d) / self.d
        return phi

    def integrate(self, f, n_points=1024):
        """Trapezoidal rule over [0,d]^2 using a uniform grid."""
        S = int(2 ** np.ceil(np.log2(max(16, int(np.sqrt(n_points))))))  # square grid
        x = np.arange(S) * self.d / S
        y = np.arange(S) * self.d / S
        X, Y = np.meshgrid(x, y, indexing='ij')
        vals = f(X, Y)
        # Trapezoid separable: integrate over x then y
        integ_x = np.trapezoid(vals, x, axis=0)
        integ = np.trapezoid(integ_x, y, axis=0)
        return np.sum(integ)  # integ already scalar, keep style

    def project_function(self, func, project_on=None, d=None):
        """
        Project a function f(x,y) onto the 2D Fourier basis via 2D FFT.
        Returns flattened coefficients ordered with kx major, ky minor.
        """
        if project_on is None:
            Lx, Ly = self.Lx, self.Ly
        else:
            Lx, Ly = project_on
        if d is None:
            d = self.d

        S = 2 ** int(np.ceil(np.log2(max(self.min_fourier_samples, 2 * max(Lx, Ly) + 1))))
        x = np.arange(S) * d / S
        y = np.arange(S) * d / S
        X, Y = np.meshgrid(x, y, indexing='ij')
        F = func(X, Y)

        c_fft = fftshift(fft2(F))
        # Scaling: (d^2 / S^2) * (1/d) = d / S^2, matching 1D style generalized
        c_fft = c_fft * (d / (S * S))

        cx0 = S // 2
        cy0 = S // 2
        x_start = cx0 - Lx
        x_end = cx0 + Lx + 1
        y_start = cy0 - Ly
        y_end = cy0 + Ly + 1
        coeffs = c_fft[x_start:x_end, y_start:y_end]  # shape (2Lx+1, 2Ly+1)
        return coeffs.reshape(-1)

    def reconstruction(self, a, X, Y):
        out = np.zeros_like(X, dtype=np.complex128)
        for c, phi in zip(a, self.phi_funcs):
            if c != 0:
                out += c * phi(X, Y)   # no extra * self.d
        return np.real(out)

    def conjugate_wrapper(self, a, c=0.0):
        """
        Convert the real-packed vector (length 2*npos) into a conjugate-symmetric complex vector (length N).
        DC set to c (default 0.0).
        """
        re = a[:self.npos]
        im = a[self.npos:]
        A = np.zeros(self.N, dtype=np.complex128)
        z = re + 1j * im
        A[self.pos_indices] = z
        A[self.neg_index_of_pos] = np.conj(z)
        A[self.zero_flat] = c
        return A

    def real_wrapper(self, a, c=0.0):
        """
        Convert conjugate-symmetric complex vector (length N) into real-packed vector (length 2*npos).
        """
        z = a[self.pos_indices]
        return np.hstack([z.real, z.imag])

    def conjugate_wrapper_matrix(self, A_real, c=0.0):
        """
        Convert a matrix whose columns are real-packed vectors into complex coefficient columns.
        """
        ncols = A_real.shape[1]
        out = np.zeros((self.N, ncols), dtype=np.complex128)
        re = A_real[:self.npos, :]
        im = A_real[self.npos:, :]
        z = re + 1j * im
        out[self.pos_indices, :] = z
        out[self.neg_index_of_pos, :] = np.conj(z)
        out[self.zero_flat, :] = c
        return out
        