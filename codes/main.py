import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
from scipy.special import beta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import time
import unittest
from fourier_utils import FourierUtils

import cProfile
import pstats
import io

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

if __name__ == '__main__':

    #unittest.main()

    def V(x):
        return (x - np.pi)**2 # - 2 * (x - np.pi)**2 + 1

    def alpha1(x):
        return np.sin(x) / np.sqrt(4 * np.pi) 
    
    def alpha2(x):
        return np.cos(x) / np.sqrt(4 * np.pi) 
    
    def alpha3(x):
        return np.sin(2*x) / np.sqrt(4 * np.pi)
    
    def alpha4(x):
        return np.cos(2*x) / np.sqrt(4 * np.pi) 
    
    def nabla_alpha1(x):
        return np.cos(x) / np.sqrt(4 * np.pi) 
    
    def nabla_alpha2(x):
        return -np.sin(x) / np.sqrt(4 * np.pi) 
    
    def nabla_alpha3(x):
        return 2 * np.cos(x) / np.sqrt(4 * np.pi) 
    
    def nabla_alpha4(x):
        return -2 * np.sin(x) / np.sqrt(4 * np.pi) 

    def W(x):
        return np.cos(x)

    def mu_0(x):
        alpha_param = 2.0
        beta_param = 2.0
        Z = (2 * np.pi)**(alpha_param + beta_param - 1) * beta(alpha_param, beta_param)
        return (x**(alpha_param - 1) * (2 * np.pi - x)**(beta_param - 1)) / Z
    
    def mu_0_mixed(x):
        alpha_param1 = 4.0
        beta_param1 = 2.0
        Z1 = (2 * np.pi)**(alpha_param1 + beta_param1 - 1) * beta(alpha_param1, beta_param1)

        alpha_param2 = 2.0
        beta_param2 = 10.0
        Z2 = (2 * np.pi)**(alpha_param2 + beta_param2 - 1) * beta(alpha_param2, beta_param2)
        return 0.5*(x**(alpha_param1 - 1) * (2 * np.pi - x)**(beta_param1 - 1)) / Z1 + 0.5*(x**(alpha_param2 - 1) * (2 * np.pi - x)**(beta_param2 - 1)) / Z2
    
    solver = McKeanVlasovSolver(L=50, d=2*np.pi, V=V, alpha=[alpha1, alpha2, alpha3, alpha4], 
                                W=W, mu_0=mu_0_mixed, min_fourier_samples=2000, delta=-0.0001, 
                                grad_alpha=[nabla_alpha1, nabla_alpha2, nabla_alpha3, nabla_alpha4], state_weight=1000)
