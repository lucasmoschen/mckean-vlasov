import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
from scipy.special import beta
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
from fourier_utils import FourierUtils, FourierUtils2D

class McKeanVlasovSolver:

    def __init__(self, L, d, V, alpha, W, mu_0, 
                 sigma=1.0, delta=0.0, M=None, grad_alpha=None, 
                 min_fourier_samples=200, state_weight=1000,
                 bar_mu_k_initial=None, final_distribution=None,
                 w_coeffs=None):
        """
        Initialize the McKean-Vlasov solver.

        Parameters:
        - L: Truncation parameter for the Fourier series.
        - d: Domain length (assuming [0, d]).
        - V: Function V(x).
        - alpha: Function alpha(x).
        - W: Function W(x).
        - mu_0: Initial density function mu_0(x).
        - sigma: Diffusion coefficient.
        - delta: Parameter delta in the equation.
        - M: Cost matrix M; if None, defaults to the identity matrix.
        - grad_alpha: Gradient of the alpha functions.
        - min_fourier_samples: Minimum number of Fourier samples.
        - state_weight: Weight for the state cost matrix.
        - bar_mu_k_initial: Initial guess for computing the bar_mu_k using Newton's method.
        - final_distribution: Final distribution to be reached. Should be the coefficients of the Fourier series.
        - w_coeffs: Coefficients for the Fourier series of W.
        """
        self.L = L
        self.d = d
        self.V = V
        self.alpha = alpha  # vector with alpha_j functions
        self.W = W
        self.mu_0 = mu_0
        self.sigma = sigma
        self.delta = delta
        self.grad_alpha = grad_alpha
        self.min_fourier_samples = min_fourier_samples

        # Instantiate Fourier helper
        self.fourier = FourierUtils(L, d, min_fourier_samples)
        self.reconstruction = self.fourier.reconstruction
        self.k_vals = self.fourier.k_vals
        self.N = self.fourier.N 
        self.phi_funcs = self.fourier.phi_funcs

        # Project functions onto Fourier basis
        self.mu0_projected = self.fourier.project_function(self.mu_0)
        assert abs(self.W(0) - self.W(d)) < 1e-10, "W must be periodic"
        assert abs(self.V(0) - self.V(d)) < 1e-10, "V must be periodic"
        if w_coeffs is None:
            self.w = self.fourier.project_function(self.W)
        else:
            self.w = w_coeffs

        # Initialize matrices
        self._compute_LV_matrix()
        self._compute_D_matrix()

        if bar_mu_k_initial is None:
            bar_mu_k_initial = np.zeros(self.N, dtype=np.complex128)
        # I should allow the code to compute all the stationary distributions here
        try:
            self.bar_mu_k = self.compute_bar_mu(method="self-consistency", 
                                                bar_mu_k_initial=bar_mu_k_initial)
        except Exception as e:
            print("WARNING - Self-consistency method failed:", e)
            self.bar_mu_k = self.compute_bar_mu(method="stationary-equation", 
                                                bar_mu_k_initial=bar_mu_k_initial[self.L+1:])

        # Control-related matrices
        if isinstance(self.grad_alpha, str) and self.grad_alpha == "constant":
            self.Psi = np.stack([2 * np.pi / self.d * 1j * np.diag(self.k_vals) for _ in range(len(alpha))], axis=0)
            self.grad_alpha = lambda x: np.full((len(alpha),) + x.shape, 1.0)
            self.alpha = [lambda x: x - self.d/2 for _ in range(len(alpha))]
        else:
            self._compute_Psi_matrix()
        self.Pi = None

        if final_distribution is None:
            self.K, self.K1 = self._compute_K_matrix(self.bar_mu_k)
            self.a0 = self.mu0_projected - self.bar_mu_k
            self.B = -np.einsum('ijk,k->ji', self.Psi, self.bar_mu_k)
        else:
            self.bar_mu_k = final_distribution
            self.K, self.K1 = self._compute_K_matrix(final_distribution)
            self.a0 = self.mu0_projected - final_distribution
            self.B = -np.einsum('ijk,k->ji', self.Psi, final_distribution)

        # Cost matrix M
        self.M = state_weight * np.eye(self.N) if M is None else M
    
    def compute_bar_mu(self, method="self-consistency", bar_mu_k_initial=None):
        if method == "self-consistency":
            if bar_mu_k_initial is None:
                bar_mu_k_initial = np.zeros(self.N, dtype=np.complex128)
            bar_mu_k = self._self_consistency(bar_mu_k_initial)
        elif method == "stationary-equation":
            if bar_mu_k_initial is None:
                bar_mu_k_initial = np.zeros(self.L, dtype=np.complex128)
            bar_mu_k = self._stationary_equation(bar_mu_k_initial)
        return bar_mu_k

    def _self_consistency(self, bar_mu_k_initial):
        """Compute the approximation of bar_mu by solving the set of non-linear equations for a complex bar_mu_k."""
        
        def equations(vector):
            bar_mu_k = vector[:self.N] + 1j * vector[self.N:]
            exponent = lambda x: -self.V(x)/self.sigma - np.sqrt(self.d)/self.sigma * sum(
                bar_mu_k[j] * self.w[j] * self.phi_funcs[j](x) for j in range(self.N)
            )
            integral_terms = self.fourier.project_function(lambda x: np.exp(exponent(x)), project_on=self.L)
            Z = self.fourier.integrate(lambda x: np.exp(exponent(x)))
            residual = integral_terms / Z - bar_mu_k
            return np.hstack([residual.real, residual.imag])

        vector_initial = np.hstack([bar_mu_k_initial.real, bar_mu_k_initial.imag])
        
        sol, _, ier, mesg = fsolve(
            equations, 
            vector_initial, 
            full_output=True
        )
        if ier != 1:
            raise ValueError("Nonlinear solver did not converge: " + mesg)
        return sol[:self.N] + 1j * sol[self.N:] 

    def _stationary_equation(self, bar_mu_k_initial):
        """
        Compute the stationary equation for bar_mu_k.
        """
        def equations(vec):
            bar_mu_k = np.hstack([
                vec[:self.L][::-1] - 1j * vec[self.L:][::-1],
                1/np.sqrt(self.d),
                vec[:self.L] + 1j * vec[self.L:]
            ])
            r = (self.L_V + self.sigma * self.D) @ bar_mu_k + self._compute_non_linear_term(bar_mu_k)
            return np.hstack([r[self.L+1:].real, r[self.L+1:].imag])

        x0 = np.hstack([bar_mu_k_initial.real, bar_mu_k_initial.imag])

        res = root(equations, x0, method='lm')
        if not res.success:
            raise ValueError("Nonlinear solver did not converge: " + res.message)
        
        sol = res.x
        bar_mu_k = np.hstack([
            sol[:self.L][::-1] - 1j * sol[self.L:][::-1],
            1/np.sqrt(self.d),
            sol[:self.L] + 1j * sol[self.L:]
        ])
        return bar_mu_k

    def _compute_LV_matrix(self):
        """
        Compute matrix L_V = <phi_j V', phi_i >.
        """
        self.L_V = np.zeros((self.N, self.N), dtype=np.complex128)
        g_coeffs = self.fourier.project_function(self.V, project_on=2 * self.L)
        l_idx_indexes = np.arange(0, 2*self.L+1)
        for k_idx in range(self.N):
            self.L_V[k_idx, :] = 4*np.pi**2*(k_idx - self.L)/self.d**2 * (k_idx - l_idx_indexes) * g_coeffs[k_idx - l_idx_indexes + 2*self.L] / np.sqrt(self.d)

    def _compute_D_matrix(self):
        """  
        Compute matrix D = <phi_j', phi_i'>.
        """
        diagonal_elements = 4 * np.pi**2 / self.d**2 * self.k_vals ** 2
        self.D = np.diag(diagonal_elements)

    def _compute_Psi_matrix(self):
        """
        Compute matrix Psi = [<phi_j alpha_j', phi_i > for j = 1,...,m]
        """
        self.Psi = []
        for alpha_j in self.alpha:
            Psi_j = np.zeros((self.N, self.N), dtype=np.complex128)
            alpha_coeffs = self.fourier.project_function(alpha_j, project_on=2 * self.L)
            l_idx_indexes = np.arange(0, 2*self.L+1)
            for k_idx in range(self.N):
                Psi_j[k_idx, :] = 4*np.pi**2*(k_idx - self.L)/self.d**2 * (k_idx - l_idx_indexes) * alpha_coeffs[k_idx - l_idx_indexes + 2*self.L] / np.sqrt(self.d)
            self.Psi.append(Psi_j)
        self.Psi = np.array(self.Psi)

    def _compute_non_linear_term(self, a):
        """
        Vectorized computation of the non-linear term:
        T_k(a,a) = (4π²/d²) * k * sum_{m=-L}^L a[k - m] * a[m] * m * w[m]
        """
        c = a * self.w * self.k_vals[::-1]
        T_sum = np.correlate(a, c, mode='same')
        return (4 * np.pi**2 / self.d**2) * self.k_vals * T_sum
    
    def _compute_K1_matrix(self, bar_mu_k):
        """
        I am writing K = K_1 + K_2 if frechet_flag is True, and K= K_1 if frechet_flag is False.
        """
        K1 = np.zeros((self.N, self.N), dtype=np.complex128)
        for k_idx in range(self.N):
            l_idx = np.arange(max(0, k_idx - self.L), min(k_idx + self.L, 2*self.L) + 1)
            diff = k_idx - l_idx
            base = 4*np.pi**2 / self.d**2 * (k_idx - self.L) * bar_mu_k[diff + self.L]
            weights = diff * self.w[diff + self.L]
            K1[k_idx, l_idx] = base * weights
        return K1
    
    def _compute_K2_matrix(self, bar_mu_k):
        """
        I am writing K = K_1 + K_2 if frechet_flag is True, and K= K_1 if frechet_flag is False.
        """
        K2 = np.zeros((self.N, self.N), dtype=np.complex128)
        for k_idx in range(self.N):
            l_idx = np.arange(max(0, k_idx - self.L), min(k_idx + self.L, 2 * self.L) + 1)
            diff = k_idx - l_idx
            base = 4*np.pi**2 / self.d**2 * (k_idx - self.L) * bar_mu_k[diff + self.L]
            weights = (l_idx - self.L) * self.w[l_idx]
            K2[k_idx, l_idx] = base * weights
        return K2

    def _compute_K_matrix(self, bar_mu_k):
        """
        Compute the matrix K = <mu_bar(W' * phi_j) + phi_j(W' * mu_bar), phi_i'>
        """
        K1 = self._compute_K1_matrix(bar_mu_k)
        K2 = self._compute_K2_matrix(bar_mu_k)
        return K1 + K2, K1
    
    def solve_riccati(self, frechet_flag=True):
        """Solve the Riccati equation to compute Pi."""
        if frechet_flag:
            A = -self.L_V - self.sigma * self.D - self.K + self.delta * np.eye(self.L_V.shape[0])
        else:
            A = -self.L_V - self.sigma * self.D - self.K1 + self.delta * np.eye(self.L_V.shape[0])
        
        mask = self.k_vals != 0
        self.check_are_conditions(A[mask][:, mask], self.B[mask, :], self.M[mask][:, mask], np.eye(self.B.shape[1]))
        self.Pi = np.zeros((self.N, self.N), dtype=np.complex128)
        self.Pi[np.ix_(mask, mask)] = solve_continuous_are(
            A[mask][:, mask], self.B[mask, :], self.M[mask][:, mask], np.eye(self.B.shape[1])
        )
        #self.check_are_conditions(A, self.B, self.M, np.eye(self.B.shape[1]))
        #self.Pi = solve_continuous_are(
        #    A, self.B, self.M, np.eye(self.B.shape[1])
        #)

    def check_are_conditions(self, A, B, Q, R):
        """
        Checks the validity of matrices A, B, Q, and R for solving the continuous ARE.
        
        Parameters:
        A : ndarray
            State transition matrix.
        B : ndarray
            Control input matrix.
        Q : ndarray
            State cost matrix (symmetric positive semi-definite).
        R : ndarray
            Control cost matrix (symmetric positive definite).
        
        Raises:
        AssertionError if any condition is not satisfied.
        """
        # Check if Q is square and symmetric
        assert Q.shape[0] == Q.shape[1], "Q must be a square matrix."
        assert np.allclose(Q, Q.T), "Q must be symmetric."

        # Check if R is square and symmetric
        assert R.shape[0] == R.shape[1], "R must be a square matrix."
        assert np.allclose(R, R.T), "R must be symmetric."

        # Check positive semi-definiteness of Q
        eigvals_Q = np.linalg.eigvals(Q)
        assert np.all(eigvals_Q >= 0), f"Q must be positive semi-definite. Eigenvalues: {eigvals_Q}"

        # Check positive definiteness of R
        eigvals_R = np.linalg.eigvals(R)
        assert np.all(eigvals_R > 0), f"R must be positive definite. Eigenvalues: {eigvals_R}"

        # Check if A and B have compatible dimensions
        assert A.shape[0] == A.shape[1], "A must be a square matrix."
        assert A.shape[0] == B.shape[0], "A and B must have compatible dimensions."

        # Check stabilizability of (A, B)
        unstable_eigenvalues = [ev for ev in np.linalg.eig(A)[0] if np.real(ev) >= 0]
        for eigenvalue in unstable_eigenvalues:
            extended_matrix = np.hstack(
                [eigenvalue * np.eye(A.shape[0]) - A, B]
            )
            rank_C = np.linalg.matrix_rank(extended_matrix)
            assert rank_C == A.shape[0], \
                    "The pair (A, B) is not stabilizable, rank(C) = {} < rank(A) = {}, eigenvalues = {}".format(rank_C, A.shape[0], unstable_eigenvalues)
        print("MESSAGE - All conditions satisfied. Matrices are suitable for solving the ARE.")

    def _solve_ode(self, ode_func, ic, t_span, t_eval, c=0.0, ode_args=()):
        """
        Helper to solve an ODE by converting between the complex formulation and
        its real representation.
        """
        real_ic = self.fourier.real_wrapper(ic, c=c) if c != 0.0 else self.fourier.real_wrapper(ic)
        sol = solve_ivp(lambda t, a: ode_func(t, a, *ode_args), t_span, real_ic,
                        t_eval=t_eval, atol=1e-10, rtol=1e-10)
        sol.y = self.fourier.conjugate_wrapper_matrix(sol.y, c=c)
        return sol

    def linearized_uncontrolled_solver(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a)
            deriv = -(self.L_V + self.sigma * self.D + self.K) @ a_complex
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval)

    def nonlinear_uncontrolled_solver_y(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D + self.K) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval)

    def nonlinear_uncontrolled_solver_mu(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a, c=1 / np.sqrt(self.d))
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv, c=1 / np.sqrt(self.d))
        return self._solve_ode(ode_system, self.mu0_projected, t_span, t_eval, c=1 / np.sqrt(self.d))

    def nonlinear_controlled_solver_mu(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a, c=1 / np.sqrt(self.d))
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D + np.einsum('ijk,i->jk', self.Psi, u(t, a_complex))) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv, c=1 / np.sqrt(self.d))
        return self._solve_ode(ode_system, self.mu0_projected, t_span, t_eval, c=1 / np.sqrt(self.d), ode_args=(u,))

    def nonlinear_controlled_solver_y(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D + self.K + np.einsum('ijk,i->jk', self.Psi, u(t, a_complex))) @ a_complex
            
            deriv += -nonlinear_term + self.B @ u(t, a_complex)
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval, ode_args=(u,))

    def linear_controlled_solver_y(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a)
            deriv = -(self.L_V + self.sigma * self.D + self.K) @ a_complex + self.B @ u(t, a_complex) + self.delta * a_complex
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval, ode_args=(u,))

    def solve_control_linearized_problem(self, t_span, t_eval=None, frechet_flag=True, compute_Pi=True):
        if compute_Pi:
            self.solve_riccati(frechet_flag)
        return self.linear_controlled_solver_y(t_span, t_eval,
            u=lambda t, a: -np.real(self.B.conj().T @ self.Pi @ a))

    def solve_control_problem(self, t_span, t_eval=None, frechet_flag=True, compute_Pi=True):
        t0 = time.time()
        if compute_Pi:
            self.solve_riccati(frechet_flag)
            print("MESSAGE - Riccati equation solved in {:.2f} seconds.".format(time.time() - t0))
        sol = self.nonlinear_controlled_solver_y(t_span, t_eval,
            u=lambda t, a: -np.real(self.B.conj().T @ self.Pi @ a))
        print("MESSAGE - Nonlinear equation solved in {:.2f} seconds.".format(time.time() - t0))
        return sol

    def weighted_L2_norm(self, a, num_points=4_096):
        """
        Compute the weighted L^2 norm ||y||_{bar_mu^{-1}} = sqrt(∫ |y(x)|^2 / bar_mu(x) dx)
        using Fourier coefficients for y and bar_mu.
        
        Parameters
        ----------
        a : array_like, shape (2L+1,)
            Fourier coefficients of y in the basis phi_k(x) = exp(i*k*x)/sqrt(2π).
        num_points : int
            Number of grid points for numerical integration (default 4096).
        
        Returns
        -------
        float
            The weighted L^2 norm ||y||_{bar_mu^{-1}}.
        """
        x = np.linspace(0, self.d, num_points, endpoint=False)
        
        phi = np.exp(1j * np.outer(self.k_vals, x)) / np.sqrt(self.d)
        y_vals = (a.conj().T @ phi)
        mu_bar_vals = np.dot(self.bar_mu_k, phi).real
        integrand = (np.abs(y_vals)**2) / mu_bar_vals[None, :]
        
        norm_squared = np.trapezoid(integrand, x, axis=1)
        return np.sqrt(norm_squared)

class McKeanVlasovSolver2D:
    """
    2D lift of the 1D McKean-Vlasov solver, keeping the same public API and style.
    L is a tuple (Lx, Ly). Domain is [0,d]^2 with the same d in both axes.
    """
    def __init__(self, L, d, V, alpha, W, mu_0,
                 sigma=1.0, delta=0.0, M=None, grad_alpha=None,
                 min_fourier_samples=200, state_weight=1000,
                 bar_mu_k_initial=None, final_distribution=None,
                 w_coeffs=None):
        self.Lx, self.Ly = L
        self.L = L  # keep a single name as in 1D
        self.d = d
        self.V = V
        self.alpha = alpha  # list of alpha_j(x,y)
        self.W = W
        self.mu_0 = mu_0
        self.sigma = sigma
        self.delta = delta
        self.grad_alpha = grad_alpha  # unused in 2D lift (per your instruction)
        self.min_fourier_samples = min_fourier_samples

        # Fourier helper
        self.fourier = FourierUtils2D(L, d, min_fourier_samples)
        self.reconstruction = self.fourier.reconstruction
        self.kx_vals = self.fourier.kx_vals
        self.ky_vals = self.fourier.ky_vals
        self.KX = self.fourier.KX
        self.KY = self.fourier.KY
        self.KX_flat = self.fourier.KX_flat
        self.KY_flat = self.fourier.KY_flat
        self.Nx = self.fourier.Nx
        self.Ny = self.fourier.Ny
        self.N = self.fourier.N
        self.phi_funcs = self.fourier.phi_funcs

        # Project initial data and potentials
        self.mu0_projected = self.fourier.project_function(self.mu_0)
        # Periodicity checks (quick corner checks)
        eps = 1e-10
        assert abs(self.W(0.0, 0.0) - self.W(self.d, 0.0)) < eps
        assert abs(self.W(0.0, 0.0) - self.W(0.0, self.d)) < eps
        assert abs(self.V(0.0, 0.0) - self.V(self.d, 0.0)) < eps
        assert abs(self.V(0.0, 0.0) - self.V(0.0, self.d)) < eps

        if w_coeffs is None:
            self.w = self.fourier.project_function(self.W)  # flattened
        else:
            self.w = w_coeffs

        # Store 2D versions for convenience
        self.w2d = self.w.reshape(self.Nx, self.Ny)

        # Matrices
        self._compute_LV_matrix()
        self._compute_D_matrix()

        if bar_mu_k_initial is None:
            bar_mu_k_initial = np.zeros(self.fourier.npos, dtype=np.complex128)

        try:
            self.bar_mu_k = self.compute_bar_mu(method="self-consistency",
                                                bar_mu_k_initial=np.zeros(self.N, dtype=np.complex128))
        except Exception as e:
            print("WARNING - Self-consistency method failed:", e)
            self.bar_mu_k = self.compute_bar_mu(method="stationary-equation",
                                                bar_mu_k_initial=bar_mu_k_initial)

        # Control-related matrices
        self._compute_Psi_matrix()
        self.Pi = None

        if final_distribution is None:
            self.K, self.K1 = self._compute_K_matrix(self.bar_mu_k)
            self.a0 = self.mu0_projected - self.bar_mu_k
            # B = - sum_j Psi_j * bar_mu (contract last index)
            self.B = -np.einsum('ijk,k->ji', self.Psi, self.bar_mu_k)
        else:
            self.bar_mu_k = final_distribution
            self.K, self.K1 = self._compute_K_matrix(final_distribution)
            self.a0 = self.mu0_projected - final_distribution
            self.B = -np.einsum('ijk,k->ji', self.Psi, final_distribution)

        # Cost matrix
        self.M = state_weight * np.eye(self.N) if M is None else M

    # ---------- Stationary distributions ----------
    def compute_bar_mu(self, method="self-consistency", bar_mu_k_initial=None):
        if method == "self-consistency":
            if bar_mu_k_initial is None:
                bar_mu_k_initial = np.zeros(self.N, dtype=np.complex128)
            return self._self_consistency(bar_mu_k_initial)
        elif method == "stationary-equation":
            if bar_mu_k_initial is None:
                bar_mu_k_initial = np.zeros(self.fourier.npos, dtype=np.complex128)
            return self._stationary_equation(bar_mu_k_initial)

    def _self_consistency(self, bar_mu_k_initial):
        """Solve bar_mu = exp(-(V + W * bar_mu)/sigma)/Z in Fourier via nonlinear equations (complex)."""
        def equations(vector):
            bar_mu_k = vector[:self.N] + 1j * vector[self.N:]
            # exponent(X,Y) = -(V + W * bar_mu)/sigma
            def exponent(X, Y):
                s = -self.V(X, Y) / self.sigma
                # sum a_k w_k φ_k
                for c, w_k, phi in zip(bar_mu_k, self.w, self.phi_funcs):
                    if c != 0 or w_k != 0:
                        s += -(np.sqrt(self.d)) * (c * w_k * phi(X, Y)).real / self.sigma  # consistent with 1D style
                return s
            integrand = lambda X, Y: np.exp(exponent(X, Y))
            integral_terms = self.fourier.project_function(lambda X, Y: np.exp(exponent(X, Y)), project_on=self.L)
            Z = self.fourier.integrate(integrand)
            residual = integral_terms / Z - bar_mu_k
            return np.hstack([residual.real, residual.imag])

        vector_initial = np.hstack([bar_mu_k_initial.real, bar_mu_k_initial.imag])
        sol, _, ier, mesg = fsolve(equations, vector_initial, full_output=True)
        if ier != 1:
            raise ValueError("Nonlinear solver did not converge: " + mesg)
        return sol[:self.N] + 1j * sol[self.N:]

    def _stationary_equation(self, bar_mu_k_initial):
        """
        Solve (L_V + sigma D) bar_mu + T(bar_mu, bar_mu) = 0
        only on the positive half-lattice (DC fixed to 1/d).
        """
        pos = self.fourier.pos_indices
        def equations(vec):
            # vec packs positive half-lattice complex values
            packed = np.hstack([vec[:len(vec)//2], vec[len(vec)//2:]])
            bar_mu = self.fourier.conjugate_wrapper(packed, c=1.0 / self.d)
            r = (self.L_V + self.sigma * self.D) @ bar_mu + self._compute_non_linear_term(bar_mu)
            r_pos = r[pos]
            return np.hstack([r_pos.real, r_pos.imag])

        x0 = np.hstack([bar_mu_k_initial.real, bar_mu_k_initial.imag])
        res = root(equations, x0, method='lm')
        if not res.success:
            raise ValueError("Nonlinear solver did not converge: " + res.message)

        sol = res.x
        bar_mu_k = self.fourier.conjugate_wrapper(np.hstack([sol[:len(sol)//2], sol[len(sol)//2:]]), c=1.0 / self.d)
        return bar_mu_k

    # ---------- Matrices ----------
    def _compute_LV_matrix(self):
        """L_V discretizes ∇·(y ∇V)."""
        self.L_V = np.zeros((self.N, self.N), dtype=np.complex128)
        Vhat = self.fourier.project_function(self.V, project_on=(2*self.Lx, 2*self.Ly)).reshape(4*self.Lx+1, 4*self.Ly+1)
        offx = 2 * self.Lx
        offy = 2 * self.Ly

        L_indices = np.arange(self.N)  # flattened l
        KX_vec = self.KX_flat[L_indices]
        KY_vec = self.KY_flat[L_indices]

        for k_idx in range(self.N):
            kx = self.KX_flat[k_idx]
            ky = self.KY_flat[k_idx]
            dKX = kx - KX_vec
            dKY = ky - KY_vec
            # pick Vhat[dKX, dKY]
            valid_x = dKX + offx
            valid_y = dKY + offy
            gh = Vhat[valid_x, valid_y]
            coeff = (4 * np.pi**2 / self.d**2) * (kx * dKX + ky * dKY) * gh / self.d  # 1/d matches 2D normalization
            self.L_V[k_idx, :] = coeff

    def _compute_D_matrix(self):
        """D discretizes Δy."""
        diag = (4 * np.pi**2 / self.d**2) * (self.KX_flat**2 + self.KY_flat**2)
        self.D = np.diag(diag)

    def _compute_Psi_matrix(self):
        """Psi_j discretizes ∇·(y ∇α_j). Shape: (m, N, N)."""
        self.Psi = []
        offx = 2 * self.Lx
        offy = 2 * self.Ly
        L_indices = np.arange(self.N)
        KX_vec = self.KX_flat[L_indices]
        KY_vec = self.KY_flat[L_indices]

        for alpha_j in self.alpha:
            Ahat = self.fourier.project_function(alpha_j, project_on=(2*self.Lx, 2*self.Ly)).reshape(4*self.Lx+1, 4*self.Ly+1)
            Psi_j = np.zeros((self.N, self.N), dtype=np.complex128)
            for k_idx in range(self.N):
                kx = self.KX_flat[k_idx]
                ky = self.KY_flat[k_idx]
                dKX = kx - KX_vec
                dKY = ky - KY_vec
                ah = Ahat[dKX + offx, dKY + offy]
                Psi_j[k_idx, :] = (4 * np.pi**2 / self.d**2) * (kx * dKX + ky * dKY) * ah / self.d
            self.Psi.append(Psi_j)
        self.Psi = np.array(self.Psi)

    # ---------- Nonlinear operator T(a,a) = ∇·( y (∇W * y) ) ----------
    def _compute_non_linear_term(self, a):
        """
        2D discrete convolution version:
        T_k(a,a) = (4π²/d²) * [ kx * (a * (a ⊙ (kx w)))_k + ky * (a * (a ⊙ (ky w)))_k ],
        where * is finite convolution (same-size, zero outside), ⊙ is pointwise.
        """
        A = a.reshape(self.Nx, self.Ny)
        kx = self.KX
        ky = self.KY
        b_x = A * (kx * self.w2d)
        b_y = A * (ky * self.w2d)

        # finite convolution, no wrap-around
        Tsum_x = convolve2d(A, b_x, mode='same', boundary='fill', fillvalue=0.0)
        Tsum_y = convolve2d(A, b_y, mode='same', boundary='fill', fillvalue=0.0)

        Tk = (4 * np.pi**2 / self.d**2) * (kx * Tsum_x + ky * Tsum_y)
        return Tk.reshape(-1)

    # ---------- Bilinearization matrices K = K1 + K2 ----------
    def _compute_K1_matrix(self, bar_mu_k):
        """
        K1[k,l] = (4π²/d²) * bar_mu_{k-l} * [ kx*(kx - lx) + ky*(ky - ly) ] * w_{k-l}
        Zero outside the truncated lattice.
        """
        K1 = np.zeros((self.N, self.N), dtype=np.complex128)
        MU = bar_mu_k.reshape(self.Nx, self.Ny)
        What = self.w.reshape(self.Nx, self.Ny)

        offx, offy = self.Lx, self.Ly
        LX, LY = self.KX_flat, self.KY_flat  # shape (N,)

        for k_idx in range(self.N):
            kx = self.KX_flat[k_idx]
            ky = self.KY_flat[k_idx]

            dKX = kx - LX
            dKY = ky - LY

            valid = (dKX >= -self.Lx) & (dKX <= self.Lx) & (dKY >= -self.Ly) & (dKY <= self.Ly)

            mu_diff = np.zeros(self.N, dtype=np.complex128)
            w_diff  = np.zeros(self.N, dtype=np.complex128)
            if np.any(valid):
                ix = (dKX[valid] + offx).astype(int)
                iy = (dKY[valid] + offy).astype(int)
                mu_diff[valid] = MU[ix, iy]
                w_diff[valid]  = What[ix, iy]

            coeff_sum = (kx * (kx - LX) + ky * (ky - LY))  # shape (N,)
            K1[k_idx, :] = (4 * np.pi**2 / self.d**2) * mu_diff * coeff_sum * w_diff

        return K1

    def _compute_K2_matrix(self, bar_mu_k):
        """
        K2[k,l] = (4π²/d²) * bar_mu_{k-l} * [ kx*lx + ky*ly ] * w_l
        Zero outside the truncated lattice for bar_mu_{k-l}.
        """
        K2 = np.zeros((self.N, self.N), dtype=np.complex128)
        MU = bar_mu_k.reshape(self.Nx, self.Ny)
        What = self.w.reshape(self.Nx, self.Ny)

        offx, offy = self.Lx, self.Ly
        LX, LY = self.KX_flat, self.KY_flat  # shape (N,)

        # w_l is always in-range since (lx,ly) ∈ [-Lx..Lx]×[-Ly..Ly]
        w_l = What[(LX + offx).astype(int), (LY + offy).astype(int)]  # shape (N,)

        for k_idx in range(self.N):
            kx = self.KX_flat[k_idx]
            ky = self.KY_flat[k_idx]

            dKX = kx - LX
            dKY = ky - LY

            valid = (dKX >= -self.Lx) & (dKX <= self.Lx) & (dKY >= -self.Ly) & (dKY <= self.Ly)

            mu_diff = np.zeros(self.N, dtype=np.complex128)
            if np.any(valid):
                ix = (dKX[valid] + offx).astype(int)
                iy = (dKY[valid] + offy).astype(int)
                mu_diff[valid] = MU[ix, iy]

            coeff_sum = (kx * LX + ky * LY)  # shape (N,)
            K2[k_idx, :] = (4 * np.pi**2 / self.d**2) * mu_diff * coeff_sum * w_l

        return K2

    def _compute_K_matrix(self, bar_mu_k):
        K1 = self._compute_K1_matrix(bar_mu_k)
        K2 = self._compute_K2_matrix(bar_mu_k)
        return K1 + K2, K1

    # ---------- Riccati ----------
    def solve_riccati(self, frechet_flag=True):
        if frechet_flag:
            A = -self.L_V - self.sigma * self.D - self.K + self.delta * np.eye(self.N)
        else:
            A = -self.L_V - self.sigma * self.D - self.K1 + self.delta * np.eye(self.N)

        mask = ~((self.KX_flat == 0) & (self.KY_flat == 0))
        self.check_are_conditions(A[mask][:, mask], self.B[mask, :], self.M[mask][:, mask], np.eye(self.B.shape[1]))
        self.Pi = np.zeros((self.N, self.N), dtype=np.complex128)
        self.Pi[np.ix_(mask, mask)] = solve_continuous_are(
            A[mask][:, mask], self.B[mask, :], self.M[mask][:, mask], np.eye(self.B.shape[1])
        )

    def check_are_conditions(self, A, B, Q, R):
        assert Q.shape[0] == Q.shape[1], "Q must be a square matrix."
        assert np.allclose(Q, Q.T), "Q must be symmetric."
        assert R.shape[0] == R.shape[1], "R must be a square matrix."
        assert np.allclose(R, R.T), "R must be symmetric."
        eigvals_Q = np.linalg.eigvals(Q); assert np.all(eigvals_Q >= 0), "Q must be PSD."
        eigvals_R = np.linalg.eigvals(R); assert np.all(eigvals_R > 0), "R must be PD."
        assert A.shape[0] == A.shape[1], "A must be square."
        assert A.shape[0] == B.shape[0], "A and B must have compatible dimensions."
        unstable = [ev for ev in np.linalg.eig(A)[0] if np.real(ev) >= 0]
        for ev in unstable:
            C = np.hstack([ev * np.eye(A.shape[0]) - A, B])
            rk = np.linalg.matrix_rank(C)
            assert rk == A.shape[0], f"(A,B) not stabilizable (rank={rk}<{A.shape[0]}), eigenvalues={unstable}"
        print("MESSAGE - All conditions satisfied. Matrices are suitable for solving the ARE.")

    # ---------- ODE helpers & solvers ----------
    def _solve_ode(self, ode_func, ic, t_span, t_eval, c=0.0, ode_args=()):
        real_ic = self.fourier.real_wrapper(ic, c=c) if c != 0.0 else self.fourier.real_wrapper(ic)
        sol = solve_ivp(lambda t, a: ode_func(t, a, *ode_args), t_span, real_ic,
                        t_eval=t_eval, atol=1e-10, rtol=1e-10)
        sol.y = self.fourier.conjugate_wrapper_matrix(sol.y, c=c)
        return sol

    def linearized_uncontrolled_solver(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a)
            deriv = -(self.L_V + self.sigma * self.D + self.K) @ a_complex
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval)

    def nonlinear_uncontrolled_solver_y(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D + self.K) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval)

    def nonlinear_uncontrolled_solver_mu(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a, c=1.0 / self.d)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv, c=1.0 / self.d)
        return self._solve_ode(ode_system, self.mu0_projected, t_span, t_eval, c=1.0 / self.d)

    def nonlinear_controlled_solver_mu(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a, c=1.0 / self.d)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D + np.einsum('ijk,i->jk', self.Psi, u(t, a_complex))) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv, c=1.0 / self.d)
        return self._solve_ode(ode_system, self.mu0_projected, t_span, t_eval, c=1.0 / self.d, ode_args=(u,))

    def nonlinear_controlled_solver_y(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_V + self.sigma * self.D + self.K + np.einsum('ijk,i->jk', self.Psi, u(t, a_complex))) @ a_complex
            deriv += -nonlinear_term + self.B @ u(t, a_complex)
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval, ode_args=(u,))

    def linear_controlled_solver_y(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a)
            deriv = -(self.L_V + self.sigma * self.D + self.K) @ a_complex + self.B @ u(t, a_complex) + self.delta * a_complex
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval, ode_args=(u,))

    def solve_control_linearized_problem(self, t_span, t_eval=None, frechet_flag=True, compute_Pi=True):
        if compute_Pi:
            self.solve_riccati(frechet_flag)
        return self.linear_controlled_solver_y(t_span, t_eval,
            u=lambda t, a: -np.real(self.B.conj().T @ self.Pi @ a))

    def solve_control_problem(self, t_span, t_eval=None, frechet_flag=True, compute_Pi=True):
        t0 = time.time()
        if compute_Pi:
            self.solve_riccati(frechet_flag)
            print("MESSAGE - Riccati equation solved in {:.2f} seconds.".format(time.time() - t0))
        sol = self.nonlinear_controlled_solver_y(t_span, t_eval,
            u=lambda t, a: -np.real(self.B.conj().T @ self.Pi @ a))
        print("MESSAGE - Nonlinear equation solved in {:.2f} seconds.".format(time.time() - t0))
        return sol

    # ---------- Norm ----------
    def weighted_L2_norm(self, a, num_points=2048):
        """
        ||y||_{bar_mu^{-1}} = sqrt( ∬ |y|^2 / bar_mu dx dy ).
        Uses separated exponentials for efficiency.
        """
        M = int(2 ** np.ceil(np.log2(max(32, int(np.sqrt(num_points))))))  # square grid
        x = np.linspace(0, self.d, M, endpoint=False)
        y = np.linspace(0, self.d, M, endpoint=False)

        Ex = np.exp(1j * (2 * np.pi / self.d) * np.outer(self.kx_vals, x))  # (Nx, M)
        Ey = np.exp(1j * (2 * np.pi / self.d) * np.outer(self.ky_vals, y))  # (Ny, M)

        A = a.reshape(self.Nx, self.Ny)
        MU = self.bar_mu_k.reshape(self.Nx, self.Ny)

        # y_vals = (1/d) * Ex^T A Ey  (shape MxM)
        y_vals = (Ex.T @ A @ Ey) / self.d
        mu_vals = np.real((Ex.T @ MU @ Ey) / self.d)

        integrand = (np.abs(y_vals) ** 2) / mu_vals
        integ_x = np.trapezoid(integrand, x, axis=0)
        integ = np.trapezoid(integ_x, y, axis=0)
        return float(np.sqrt(np.sum(integ)))

def build_example(L=8, d=2*np.pi, K=1.0, sigma=0.5):
    # Allow either int (square truncation) or tuple (Lx, Ly)
    L_tuple = (int(L), int(L)) if np.isscalar(L) else (int(L[0]), int(L[1]))

    V = lambda X, Y: np.zeros_like(X)
    W = lambda x, y: K * (np.cos(x) + np.cos(y))
    alpha_list = [
        lambda x, y: np.cos(x),
        lambda x, y: np.sin(x),
        lambda x, y: np.cos(y),
        lambda x, y: np.sin(y),
    ]
    def mu1d(theta):
        return (1.0/d) * (1.0 + 0.2*np.cos(theta) + 0.1*np.sin(theta))
    mu0 = lambda x, y: mu1d(x) * mu1d(y)
    return McKeanVlasovSolver2D(L_tuple, d, V, alpha_list, W, mu0, sigma=sigma)


def sanity_checks(L=8, d=2*np.pi, K=1.0, sigma=0.5, T=2.0, steps=20):
    sol = build_example(L=L, d=d, K=K, sigma=sigma)
    t_eval = np.linspace(0.0, T, steps + 1)
    out = sol.nonlinear_uncontrolled_solver_mu((0.0, T), t_eval=t_eval)

    Nx, Ny = sol.Nx, sol.Ny
    Lx, Ly = sol.L
    dside = sol.d

    mass_err, int_err, conj_err, rank1_err, axis_frac = [], [], [], [], []

    x = np.linspace(0, dside, 256, endpoint=False)
    y = np.linspace(0, dside, 256, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    for i in range(len(t_eval)):
        # out.y[:, i] is a complex flattened column of length Nx*Ny
        A = out.y[:, i].reshape(Nx, Ny)

        # DC / mass checks
        mass_err.append(abs(A[Lx, Ly] - 1.0 / dside))

        # reconstruct μ on a dense grid
        mu_vals = sol.reconstruction(A.reshape(-1), X, Y)
        integ = float(np.trapezoid(np.trapezoid(mu_vals, x, axis=0), y, axis=0))
        int_err.append(abs(integ - 1.0))

        # conjugate symmetry (A[k] = conj A[-k])
        Aflip = np.flip(np.flip(A.conj(), axis=0), axis=1)
        conj_err.append(float(np.max(np.abs(A - Aflip))))

        # rank-1 separability via SVD on coeff grid
        _, S, _ = np.linalg.svd(A, full_matrices=False)
        rel = float((S[1:].sum()) / (S.sum() + 1e-15))
        rank1_err.append(rel)

        # axis energy fraction (kx=0 row and ky=0 column; beware double-counting DC)
        tot = float(np.sum(np.abs(A)**2))
        axes = float(np.sum(np.abs(A[Lx, :])**2) + np.sum(np.abs(A[:, Ly])**2) - abs(A[Lx, Ly])**2)
        axis_frac.append(axes / (tot + 1e-15))

    return {
        't': t_eval,
        'mass_err_max': float(np.max(mass_err)),
        'integral_err_max': float(np.max(int_err)),
        'conjugate_symmetry_err_max': float(np.max(conj_err)),
        'rank1_rel_error_max': float(np.max(rank1_err)),
        'axis_energy_fraction_final': float(axis_frac[-1]),
        'axis_energy_fraction_min': float(np.min(axis_frac)),
    }

def compare_field_difference(L=8, d=2*np.pi, K=1.0, sigma=0.5, T=2.0, steps=60, solver_1d_cls=None):
    """
    Compare μ_2D(x,y) from the 2D solver with ρ1(x)ρ2(y) built from two 1D runs.
    Uses your 1D class' nonlinear_uncontrolled_solver_mu.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert solver_1d_cls is not None, "Pass your 1D McKeanVlasovSolver class via solver_1d_cls"

    # 2D run (V=0, W=K(cos x + cos y), separable μ0)
    sol2 = build_example(L=L, d=d, K=K, sigma=sigma)
    t_eval = np.linspace(0.0, T, steps + 1)
    out2 = sol2.nonlinear_uncontrolled_solver_mu((0.0, T), t_eval=t_eval)
    Nx, Ny = sol2.Nx, sol2.Ny
    A2_final = out2.y[:, -1].reshape(Nx, Ny)

    # dense grid for visualization
    x = np.linspace(0, d, 300, endpoint=False)
    y = np.linspace(0, d, 300, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    mu2_final = sol2.reconstruction(A2_final.reshape(-1), X, Y)

    # Helper: reconstruct 1D from complex coeffs (basis 1/√d)
    def reconstruct_1d_field(a_complex, xgrid, d_len, k_vals):
        return np.real(np.exp(2j * np.pi * np.outer(k_vals, xgrid) / d_len).T @ a_complex / np.sqrt(d_len))

    # 1D data
    V1d = lambda s: np.zeros_like(s)
    W1d = lambda s: K * np.cos(s)
    alpha1d = [lambda s: np.cos(s), lambda s: np.sin(s)]  # not used for uncontrolled
    def mu1d(theta): return (1.0/d) * (1.0 + 0.2*np.cos(theta) + 0.1*np.sin(theta))

    # X-axis 1D
    sol1_x = solver_1d_cls(L if np.isscalar(L) else int(L[0]), d, V1d, alpha1d, W1d, mu1d, sigma=sigma)
    out1x = sol1_x.nonlinear_uncontrolled_solver_mu((0.0, T), t_eval=t_eval)
    a1x_final = out1x.y[:, -1]  # complex (2L+1,)
    rho1_vals = reconstruct_1d_field(a1x_final, x, d, sol1_x.k_vals)

    # Y-axis 1D (same L if scalar, else use Ly)
    sol1_y = solver_1d_cls(L if np.isscalar(L) else int(L[1]), d, V1d, alpha1d, W1d, mu1d, sigma=sigma)
    out1y = sol1_y.nonlinear_uncontrolled_solver_mu((0.0, T), t_eval=t_eval)
    a1y_final = out1y.y[:, -1]
    rho2_vals = reconstruct_1d_field(a1y_final, y, d, sol1_y.k_vals)

    # Product vs 2D
    mu_prod = rho1_vals[:, None] * rho2_vals[None, :]
    diff_field = mu2_final - mu_prod
    l2_rel = np.linalg.norm(diff_field) / (np.linalg.norm(mu2_final) + 1e-15)

    # Plots
    plt.figure(figsize=(6, 5))
    pcm = plt.pcolormesh(X, Y, diff_field, shading='auto', cmap='RdBu')
    plt.colorbar(pcm, label=r"$\mu_{2D}(x,y) - \rho_1(x)\rho_2(y)$")
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Difference at final time T')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(7, 4))
    y0 = 0
    plt.plot(x, mu2_final[:, y0], label=r"$\mu_{2D}(x,0)$")
    plt.plot(x, rho1_vals * rho2_vals[y0], '--', label=r"$\rho_1(x)\rho_2(0)$")
    plt.xlabel('x'); plt.title('Cut at y=0'); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7, 4))
    x0 = 0
    plt.plot(y, mu2_final[x0, :], label=r"$\mu_{2D}(0,y)$")
    plt.plot(y, rho2_vals * rho1_vals[x0], '--', label=r"$\rho_2(y)\rho_1(0)$")
    plt.xlabel('y'); plt.title('Cut at x=0'); plt.legend(); plt.tight_layout(); plt.show()

    return {'rel_L2_error': float(l2_rel)}

if __name__ == '__main__':

    print(sanity_checks())
    print(compare_field_difference(solver_1d_cls=McKeanVlasovSolver))