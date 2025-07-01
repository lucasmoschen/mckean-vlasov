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

    def __init__(self, L, d, G, alpha, W, mu_0, 
                 sigma=1.0, delta=0.0, M=None, grad_alpha=None, 
                 min_fourier_samples=200, state_weight=1000,
                 bar_mu_k_initial=None, final_distribution=None,
                 w_coeffs=None):
        """
        Initialize the McKean-Vlasov solver.

        Parameters:
        - L: Truncation parameter for the Fourier series.
        - d: Domain length (assuming [0, d]).
        - G: Function G(x).
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
        self.G = G
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
        assert abs(self.G(0) - self.G(d)) < 1e-10, "G must be periodic"
        if w_coeffs is None:
            self.w = self.fourier.project_function(self.W)
        else:
            self.w = w_coeffs

        # Initialize matrices
        self._compute_LG_matrix()
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
            exponent = lambda x: -self.G(x)/self.sigma - np.sqrt(self.d)/self.sigma * sum(
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
            r = (self.L_G + self.sigma * self.D) @ bar_mu_k + self._compute_non_linear_term(bar_mu_k)
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

    def _compute_LG_matrix(self):
        """
        Compute matrix L_G = <phi_j G', phi_i >.
        """
        self.L_G = np.zeros((self.N, self.N), dtype=np.complex128)
        g_coeffs = self.fourier.project_function(self.G, project_on=2 * self.L)
        l_idx_indexes = np.arange(0, 2*self.L+1)
        for k_idx in range(self.N):
            self.L_G[k_idx, :] = 4*np.pi**2*(k_idx - self.L)/self.d**2 * (k_idx - l_idx_indexes) * g_coeffs[k_idx - l_idx_indexes + 2*self.L] / np.sqrt(self.d)

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
            A = -self.L_G - self.sigma * self.D - self.K + self.delta * np.eye(self.L_G.shape[0])
        else:
            A = -self.L_G - self.sigma * self.D - self.K1 + self.delta * np.eye(self.L_G.shape[0])
        
        self.check_are_conditions(A, self.B, self.M, np.eye(self.B.shape[1]))

        self.Pi = solve_continuous_are(
            A, self.B, self.M, np.eye(self.B.shape[1])
        )

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
            deriv = -(self.L_G + self.sigma * self.D + self.K) @ a_complex + self.delta * a_complex
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval)

    def nonlinear_uncontrolled_solver_y(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_G + self.sigma * self.D + self.K) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval)

    def nonlinear_uncontrolled_solver_mu(self, t_span, t_eval=None):
        def ode_system(t, a, *args):
            a_complex = self.fourier.conjugate_wrapper(a, c=1 / np.sqrt(self.d))
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_G + self.sigma * self.D) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv, c=1 / np.sqrt(self.d))
        return self._solve_ode(ode_system, self.mu0_projected, t_span, t_eval, c=1 / np.sqrt(self.d))

    def nonlinear_controlled_solver_mu(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a, c=1 / np.sqrt(self.d))
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_G + self.sigma * self.D + np.einsum('ijk,i->jk', self.Psi, u(t, a_complex))) @ a_complex - nonlinear_term
            return self.fourier.real_wrapper(deriv, c=1 / np.sqrt(self.d))
        return self._solve_ode(ode_system, self.mu0_projected, t_span, t_eval, c=1 / np.sqrt(self.d), ode_args=(u,))

    def nonlinear_controlled_solver_y(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a)
            nonlinear_term = self._compute_non_linear_term(a_complex)
            deriv = -(self.L_G + self.sigma * self.D + self.K + np.einsum('ijk,i->jk', self.Psi, u(t, a_complex))) @ a_complex
            
            deriv += -nonlinear_term + self.B @ u(t, a_complex)
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval, ode_args=(u,))

    def linear_controlled_solver_y(self, t_span, t_eval=None, u=None):
        def ode_system(t, a, u):
            a_complex = self.fourier.conjugate_wrapper(a)
            deriv = -(self.L_G + self.sigma * self.D + self.K) @ a_complex + self.B @ u(t, a_complex) + self.delta * a_complex
            return self.fourier.real_wrapper(deriv)
        return self._solve_ode(ode_system, self.a0, t_span, t_eval, ode_args=(u,))

    def solve_control_linearized_problem(self, t_span, t_eval=None, frechet_flag=True):
        if self.Pi is None:
            self.solve_riccati(frechet_flag)
        return self.linear_controlled_solver_y(t_span, t_eval,
            u=lambda t, a: -np.real(self.B.conj().T @ self.Pi @ a))

    def solve_control_problem(self, t_span, t_eval=None, frechet_flag=True):
        t0 = time.time()
        if self.Pi is None:
            self.solve_riccati(frechet_flag)
            print("MESSAGE - Riccati equation solved in {:.2f} seconds.".format(time.time() - t0))
        sol = self.nonlinear_controlled_solver_y(t_span, t_eval,
            u=lambda t, a: -np.real(self.B.conj().T @ self.Pi @ a))
        print("MESSAGE - Nonlinear equation solved in {:.2f} seconds.".format(time.time() - t0))
        return sol

class TestImpromentFunction(unittest.TestCase):
    
    def setUp(self):
        """
        Describe the set ups your experiment with self.
        """

        def G(x):
            return (x - np.pi)**2

        def alpha(x):
            return np.sin(x)

        def W(x):
            return (x - np.pi)**2

        def mu_0(x):
            alpha_param = 2.0
            beta_param = 2.0
            Z = (2 * np.pi)**(alpha_param + beta_param - 1) * beta(alpha_param, beta_param)
            return (x**(alpha_param - 1) * (2 * np.pi - x)**(beta_param - 1)) / Z

        self.model = McKeanVlasovSolver(L=30, d=2*np.pi, G=G, alpha=alpha, W=W, mu_0=mu_0, min_fourier_samples=2000)
        self.a = np.random.random(self.model.N) + 1j * np.random.random(self.model.N)

    def test_performance_comparison(self):
        # Measure performance of the original function
        start_time_original = time.time()
        result_original = self.original_function(self.a)
        time_original = time.time() - start_time_original

        # Measure performance of the improved function
        start_time_improved = time.time()
        result_improved = self.new_function(self.a)
        time_improved = time.time() - start_time_improved

        # Output the times for comparison
        print(f"Original function time: {time_original} seconds")
        print(f"Improved function time: {time_improved} seconds")

        # Optionally, assert that improved is faster than original
        self.assertTrue(time_improved < time_original, "The improved function is not faster than the original")

        # Ensure the results from both functions are the same
        np.testing.assert_array_almost_equal(result_original, result_improved, decimal=5)

    def test_method_agreement(self):
        # Compute results from both functions
        result_original = self.original_function(self.a)
        result_improved = self.new_function(self.a)

        # Use assertArrayAlmostEqual from numpy.testing to compare arrays
        np.testing.assert_array_almost_equal(result_original, result_improved, decimal=6,
                                             err_msg="The outputs of the original and improved functions do not match.")

class McKeanVlasovPlotter:
    
    def __init__(self, solver):
        self.solver = solver

    def plot_function_over_time(self, x, data, times, ylabel, title):
        """General function for plotting various data over time using a colorblind-friendly palette."""
        # Define a set of colorblind-friendly colors
        colors = [
            "#E69F00",  # Orange
            "#56B4E9",  # Sky Blue
            "#009E73",  # Bluish Green
            "#F0E442",  # Yellow
            "#0072B2",  # Blue
            "#D55E00",  # Vermillion
            "#CC79A7"   # Reddish Purple
        ]

        fig, ax = plt.subplots()
        for i, time in enumerate(times):
            # Cycle through colors if there are more times than colors
            color = colors[i % len(colors)]
            ax.plot(x, data[:, i], label=f't={time:.2f}', color=color)

        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend()
        ax.set_title(title)
        plt.show()

    def plot_mu_x_t(self, t_values):
        x = np.linspace(0, self.solver.d, 1000)
        solution = self.solver.nonlinear_uncontrolled_solver_mu(t_span=(0, max(t_values)), t_eval=t_values)
        t_indices = [np.argmin(np.abs(solution.t - t)) for t in t_values]
        data = np.array([self.solver.reconstruction(solution.y[:, i], x) for i in t_indices]).T
        self.plot_function_over_time(x, data, solution.t[t_indices], '$\mu(x, t)$', 'Nonlinear Uncontrolled $\mu(x, t)$ for different times')

    def plot_y_x_t(self, t_values):
        x = np.linspace(0, self.solver.d, 1000)
        solution = self.solver.nonlinear_uncontrolled_solver_y(t_span=(0, max(t_values)), t_eval=t_values)
        t_indices = [np.argmin(np.abs(solution.t - t)) for t in t_values]
        data = np.array([self.solver.reconstruction(solution.y[:, i], x) for i in t_indices]).T
        self.plot_function_over_time(x, data, solution.t[t_indices], '$y(x, t)$', 'Nonlinear Uncontrolled $y(x, t)$ for different times')

    def plot_mu_bar_x(self, name=None):
        x = np.linspace(0, self.solver.d, 1000)
        mu_bar = self.solver.reconstruction(self.solver.bar_mu_k, x)
        mu0 = self.solver.reconstruction(self.solver.mu0_projected, x)
        plt.figure(figsize=(6,3))
        plt.plot(x, mu_bar, label=r'$\bar{\mu}(x)$')
        plt.plot(x, mu0, label=r'$\mu_0(x)$')
        plt.xlabel('$x$', fontsize=14)
        plt.title(r'Plot of $\bar{\mu}(x)$ and $\mu_0(x)$')
        plt.legend()
        plt.tight_layout()

        if name is not None:
            plt.savefig(name, format='PDF', bbox_inches='tight')
        else:
            plt.show()

    def plot_control_and_norm(self, t_max, name=None):
        # Generate the control function values
        solution = self.solver.solve_control_problem(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(500, int(np.ceil(t_max * 100)))))
        solution2 = self.solver.nonlinear_uncontrolled_solver_y(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(500, int(np.ceil(t_max * 100)))))
        t_points = solution.t
        control = np.array([-np.real(self.solver.B.conj().T @ self.solver.Pi @ solution.y[:, i]) for i in range(len(t_points))]).T

        # Calculate the L^2 norm of y(t)
        y_norm = np.linalg.norm(solution.y, axis=0)
        y_norm2 = np.linalg.norm(solution2.y, axis=0)

        # Creating the subplot figure
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Plotting ||y(., t)||_{L^2} over time
        ax1.plot(t_points, y_norm, color="#0072B2", label="Controlled")
        ax1.plot(t_points, y_norm2, color="red", label="Uncontrolled")
        ax1.set_xlabel('Time $t$', fontsize=14)
        ax1.set_ylabel('$||y(., t)||_{L^2}$', fontsize=14)
        ax1.set_title('Norm of $y(t)$ over Time', fontsize=18)
        ax1.set_yscale("log")
        ax1.legend(fontsize=15)

        # Plotting the control function
        for j in range(len(self.solver.alpha)):
            ax2.plot(t_points, abs(control[j]), label=r"$\alpha_{}$".format(j+1))
        ax2.set_xlabel('Time $t$', fontsize=14)
        ax2.set_ylabel('Control $u(t)$', fontsize=14)
        ax2.set_title(r'Control Functions over Time for each $\alpha$', fontsize=18)
        ax2.set_yscale("log")
        ax2.legend(fontsize=15)

        # Display the plots
        plt.tight_layout()
        if name is not None:
            plt.savefig(name, format='PDF', bbox_inches='tight')
        else:
            plt.show()

    def plot_control_and_norm_linear(self, t_max):
        # Generate the control function values
        solution = self.solver.solve_control_linearized_problem(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(500, int(np.ceil(t_max * 100)))))
        solution2 = self.solver.linearized_uncontrolled_solver(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(500, int(np.ceil(t_max * 100)))))
        t_points = solution.t
        control = np.array([-np.real(self.solver.B.conj().T @ self.solver.Pi @ solution.y[:, i]) for i in range(len(t_points))]).T

        # Calculate the L^2 norm of y(t)
        y_norm = np.linalg.norm(solution.y, axis=0)
        y_norm2 = np.linalg.norm(solution2.y, axis=0)

        # Creating the subplot figure
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plotting ||y(., t)||_{L^2} over time
        ax1.plot(t_points, y_norm, color="#0072B2", label="Controlled")
        ax1.plot(t_points, y_norm2, color="red", label="Uncontrolled")
        ax1.set_yscale("log")
        ax1.set_xlabel('Time $t$', fontsize=14)
        ax1.set_ylabel('$||y(., t)||_{L^2}$', fontsize=14)
        ax1.set_title('Norm of $y(t)$ linearized over Time', fontsize=18)
        ax1.legend()

        # Plotting the control function
        for j in range(len(self.solver.alpha)):
            ax2.plot(t_points, abs(control[j]), label=r"$\alpha_{}$".format(j+1))
        ax2.set_yscale("log")
        ax2.set_xlabel('Time $t$', fontsize=14)
        ax2.set_ylabel('Control $u(t)$', fontsize=14)
        ax2.set_title('Control Functions over Time', fontsize=18)
        ax2.legend()

        # Display the plots
        plt.tight_layout()
        plt.show()

    def plot_y_diff_L2_norm(self, t_max):

        y_nonlinear = self.solver.nonlinear_uncontrolled_solver_y(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(500, int(np.ceil(t_max * 30)))))
        y_linear = self.solver.linearized_uncontrolled_solver(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(500, int(np.ceil(t_max * 30)))))

        # Calculate the L2 norm difference
        diff = np.linalg.norm(y_nonlinear.y - y_linear.y, axis=0)
        plt.figure()
        plt.plot(y_linear.t, diff)
        plt.xlabel('Time $t$', fontsize=14)
        plt.ylabel('$||y(., t) - y_L(., t)||_{L^2}$', fontsize=14)
        plt.title(f'L2 norm difference over time')
        plt.show()

    def plot_pi_matrix(self, cmap='viridis'):

        pi_abs = np.minimum(np.abs(self.solver.Pi), 1)

        plt.figure(figsize=(8, 6))
        sns.heatmap(pi_abs, fmt=".2f", cmap=cmap)
        
        plt.title('Absolute Values of Pi Matrix')
        plt.tight_layout()
        plt.show()

        pi_abs = np.minimum(np.abs(self.solver.Pi @ self.solver.B), 10)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pi_abs, fmt=".2f", cmap=cmap)
        
        plt.title('Absolute Values of B^* Pi Matrix')
        plt.tight_layout()
        plt.show()

    def animate_solution(self, t_values):
        # Precompute the solution for the given time values
        t_span = (0, max(t_values))
        solution = self.solver.solve_control_problem(t_span=t_span, t_eval=t_values)
        x = np.linspace(0, self.solver.d, 1000)
        grad_alpha_x = [self.solver.grad_alpha[j](x) for j in range(self.alpha)]  # Evaluate alpha over x
        y_reconstructed = [self.solver.reconstruction(solution.y[:, i], x) for i in range(len(solution.t))]
        
        # Calculate overall min and max from reconstructed values and the alpha*control product
        min_y = min([np.min(y) for y in y_reconstructed])
        max_y = max([np.max(y) for y in y_reconstructed])
        
        # Compute control values
        controls = [-self.solver.B.conj().T @ self.solver.Pi @ solution.y[:, i] for i in range(len(solution.t))]
        alpha_control = [np.sum([grad_alpha_x[j] * control[j] for j in range(self.solver.alpha)], axis=0) for control in controls]
        
        # Adjust the y-limits to include alpha*control plots
        min_y = min(min_y, min(np.min(ac) for ac in alpha_control))
        max_y = max(max_y, max(np.max(ac) for ac in alpha_control))

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2, label='y(x, t)')
        line_alpha_control, = ax.plot([], [], lw=2, label=r'$\nabla \alpha(x) \cdot u(t)$', linestyle='--', color='red')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ax.set_title("Dynamics of y over Time")
        ax.legend()

        def init():
            line.set_data([], [])
            line_alpha_control.set_data([], [])
            time_text.set_text('')
            ax.set_xlim(0, self.solver.d)
            ax.set_ylim(min_y, max_y)
            return line, line_alpha_control, time_text

        def update(frame):
            y = y_reconstructed[frame]
            line.set_data(x, y)
            ac = alpha_control[frame]
            line_alpha_control.set_data(x, ac)
            time_text.set_text(f'Time = {t_values[frame]:.2f}s')
            return line, line_alpha_control, time_text

        ani = FuncAnimation(fig, update, frames=len(t_values), init_func=init, blit=True, interval=200)
        plt.show()

def profile_time_analyser(solver):
    
    profiler = cProfile.Profile()
    profiler.enable()

    def profile_solver(solver):
        # Define here the function you would like to analyse.
        t_max = 10
        solution = solver.solve_control_problem(t_span=(0, t_max), t_eval=np.linspace(0, t_max, max(100, int(np.ceil(t_max * 30)))))

    profile_solver(solver)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

if __name__ == '__main__':

    #unittest.main()

    def G(x):
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
    
    solver = McKeanVlasovSolver(L=50, d=2*np.pi, G=G, alpha=[alpha1, alpha2, alpha3, alpha4], 
                                W=W, mu_0=mu_0_mixed, min_fourier_samples=2000, delta=-0.0001, 
                                grad_alpha=[nabla_alpha1, nabla_alpha2, nabla_alpha3, nabla_alpha4], state_weight=1000)

    #profile_time_analyser(solver)

    plotter = McKeanVlasovPlotter(solver)

    #plotter.plot_mu_bar_x()

    plotter.plot_control_and_norm(t_max=5.0)

    #plotter.plot_control_and_norm(t_max=0.5)

    #plotter.plot_pi_matrix()

    #plotter.plot_y_diff_L2_norm(t_max=0.5)

    #plotter.animate_solution(t_values=np.linspace(0, 0.5, 50))