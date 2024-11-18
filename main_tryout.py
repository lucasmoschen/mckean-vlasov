import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import fsolve
from scipy.special import beta
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import time
import unittest
from scipy.linalg import eigvals

class McKeanVlasovSolver:

    def __init__(self, L, d, G, alpha, W, mu_0, sigma=1.0, delta=0.0, M=None, grad_alpha=None, min_fourier_samples=200):
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
        """
        self.L = L
        self.d = d
        self.G = G
        self.alpha = alpha
        self.W = W
        self.mu_0 = mu_0
        self.sigma = sigma
        self.delta = delta

        # Define the Fourier basis functions
        self.k_vals = np.arange(-L, L + 1)
        self.N = len(self.k_vals)  # Total number of modes

        # Compute the orthonormal Fourier basis functions
        self.phi_funcs = [self._phi_k(k) for k in self.k_vals]

        # Project mu_0 onto the Fourier basis
        self.mu0_projected = self._project_mu0(min_fourier_samples)
        # Project W onto the Fourier basis
        self.w = self._project_Fourier_basis_FFT(self.W, min_fourier_samples)
        # Initialize matrices
        self._compute_LG_matrix(min_fourier_samples)
        self._compute_D_matrix()
        try:
            self.compute_bar_mu()
        except:
            self.bar_mu_k = self.compute_bar_mu_method2()
        self._compute_K_matrix()
        # Project y0 onto the Fourier basis
        self.a0 = self.bar_mu_k - self.mu0_projected

        # Control-related matrices
        self._compute_Psi_matrix(min_fourier_samples)
        self.b = self.Psi @ self.bar_mu_k
        self.Pi = None

        # Cost matrix M
        self.M = np.identity(self.N) if M is None else M

    def reconstruction(self, a, x):
        """
        Reconstruct a function from coefficients a and points x.
        """
        return np.sum([a[k]*self.phi_funcs[k](x) for k in range(self.N)], axis=0)

    def _phi_k(self, k):
        """Return the k-th Fourier basis function, scaled to be orthonormal over [0, d]."""
        return lambda x: np.exp(2*np.pi*1j*k*x/self.d) / np.sqrt(self.d)

    def _project_mu0(self, min_fourier_samples=200):
        """Project the initial density mu_0 onto the Fourier basis."""
        return self._project_Fourier_basis_FFT(self.mu_0, min_fourier_samples)
    
    def _project_Fourier_basis_FFT(self, func, min_fourier_samples=200, project_on=None):
        """
        Project the function func onto the Fourier basis.
        - func: the function to be projected. 
        - min_fourier_samples: the number of samples to approximate the integral.
        """
        if project_on is None:
            project_on = self.L

        c = np.zeros(2*project_on+1, dtype=np.complex128)
        samples = 2**int(np.ceil(np.log2(max(min_fourier_samples, self.L + 2, project_on + 2))))
        f = func((np.linspace(0, samples, samples, endpoint=False) + 0.5)*self.d/samples)
        c_fft = (np.sqrt(self.d)/samples)*fft(f)*np.exp(-np.pi*1j*np.linspace(0, samples, samples, endpoint=False)/samples)
        c[project_on:] = c_fft[0: project_on+1]
        c[0:project_on] = np.conjugate(c_fft[1:project_on+1][::-1])
        return c

    def _integrate(self, f):
        """Numerical integration over [0, d] using the trapezoidal rule."""
        x_vals = np.linspace(0, self.d, 1000)
        y_vals = f(x_vals)
        return np.trapezoid(y_vals, x_vals)

    def _grad_finite_differences(self, F):
        """
        Compute the gradienf of the function F and returns a function.
        """
        h = 1e-6
        return lambda x: (F(x + h) - F(x - h)) / (2 * h)

    def _convolve_W_phi(self, phi_j):
        """Compute the convolution W * phi_j."""
        def convolved_function(x):
            result = np.zeros_like(x, dtype=np.complex128)
            y_vals = np.linspace(0, self.d, 10000)
            for idx, x_val in enumerate(x):
                W_vals = self.W((x_val - y_vals) % self.d)
                phi_vals = phi_j(y_vals)
                integrand = W_vals * phi_vals
                result[idx] = np.trapezoid(integrand, y_vals)
            return result
        return convolved_function

    def compute_bar_mu(self, min_fourier_samples=200):
        """Compute the approximation of bar_mu by solving the set of non-linear equations for a complex bar_mu_k."""
        
        def equations(vector):
            bar_mu_k = vector[:self.N] + 1j * vector[self.N:]
            exponent = lambda x: -self.G(x)/self.sigma - np.sqrt(self.d)/self.sigma * sum(
                bar_mu_k[j] * self.w[j] * self.phi_funcs[j](x) for j in range(self.N)
            )
            integral_terms = self._project_Fourier_basis_FFT(lambda x: np.exp(exponent(x)), min_fourier_samples)
            Z = self._integrate(lambda x: np.exp(exponent(x)))
            residual = integral_terms / Z - bar_mu_k
            residuals = np.hstack([residual.real, residual.imag])
            return residuals

        # Initial guess: start with zeros for both real and imaginary parts
        bar_mu_k_initial = np.zeros(self.N, dtype=np.complex128)
        vector_initial = np.hstack([bar_mu_k_initial.real, bar_mu_k_initial.imag])
        
        sol, _, ier, mesg = fsolve(
            equations, 
            vector_initial, 
            full_output=True
        )
        
        if ier != 1:
            raise ValueError("Nonlinear solver did not converge: " + mesg)
        
        self.bar_mu_k = sol[:self.N] + 1j * sol[self.N:] 

    def compute_bar_mu_method2(self):
        """Compute the approximation of bar_mu by solving the set of non-linear equations for a complex bar_mu_k."""
        
        def equations(vector):
            bar_mu_k = np.hstack([vector[:self.L][::-1] - 1j * vector[self.L:][::-1], 1/np.sqrt(self.d), vector[:self.L] + 1j * vector[self.L:]])
            residual = (self.L_G + self.sigma * self.D) @ bar_mu_k + self._compute_non_linear_term(bar_mu_k)
            residuals = np.hstack([residual[self.L+1:].real, residual[self.L+1:].imag])
            return residuals

        # Initial guess: start with zeros for both real and imaginary parts
        bar_mu_k_initial = np.zeros(self.L, dtype=np.complex128)
        vector_initial = np.hstack([bar_mu_k_initial.real, bar_mu_k_initial.imag])
        
        sol, _, ier, mesg = fsolve(
            equations, 
            vector_initial, 
            full_output=True
        )
        
        if ier != 1:
            raise ValueError("Nonlinear solver did not converge: " + mesg)
        
        bar_mu_k = np.hstack([sol[:self.L][::-1] - 1j * sol[self.L:][::-1], 1/np.sqrt(self.d), sol[:self.L] + 1j * sol[self.L:]])
        return bar_mu_k

    def _compute_LG_matrix(self, min_fourier_samples):
        """
        Compute matrix L_G = <phi_j G', phi_i >.
        """
        self.L_G = np.zeros((self.N, self.N), dtype=np.complex128)
        g_coeffs = self._project_Fourier_basis_FFT(self.G, project_on=2*self.L, min_fourier_samples=min_fourier_samples)
        l_idx_indexes = np.arange(0, 2*self.L+1)
        for k_idx in range(self.N):
            self.L_G[k_idx, :] = 4*np.pi**2*(k_idx - self.L)/self.d**2 * (k_idx - l_idx_indexes) * g_coeffs[k_idx - l_idx_indexes + 2*self.L] / np.sqrt(self.d)

    def _compute_D_matrix(self):
        """
        Compute matrix D = <phi_j', phi_i'>.
        """
        diagonal_elements = 4 * np.pi**2 / self.d**2 * self.k_vals ** 2
        self.D = np.diag(diagonal_elements)

    def _compute_Psi_matrix(self, min_fourier_samples):
        """
        Compute matrix Psi = <phi_j alpha', phi_i >.
        """
        self.Psi = np.zeros((self.N, self.N), dtype=np.complex128)
        alpha_coeffs = self._project_Fourier_basis_FFT(self.alpha, project_on=2*self.L, min_fourier_samples=min_fourier_samples)
        l_idx_indexes = np.arange(0, 2*self.L+1)
        for k_idx in range(self.N):
            self.Psi[k_idx, :] = 4*np.pi**2*(k_idx - self.L)/self.d**2 * (k_idx - l_idx_indexes) * alpha_coeffs[k_idx - l_idx_indexes + 2*self.L] / np.sqrt(self.d)

    def _compute_T(self):
        """
        Compute the tensor T_{i,k,l}.
        Notice that this matrix is very sparse. Only N^2 elements are non zero.
        """

        self.T = np.zeros((self.N, self.N, self.N), dtype=np.complex128)
        for k_idx in range(self.N):
            for l_idx in range(self.N):
                for m_idx in range(self.N):
                    if m_idx == k_idx - l_idx and abs(k_idx - l_idx) <= self.L:
                        self.T[k_idx, l_idx, m_idx] = 4*np.pi**2 / self.d**2 * (k_idx-self.L) * (m_idx - self.L) * self.w[m_idx]

    def _compute_non_linear_term_old(self, a):
        """
        Computes sum_j sum_k a_j a_k T_ijk using the structure of T.
        """
        non_linear_term = np.zeros(self.N, dtype=np.complex128)
        for k_idx in range(self.N):
            for l_idx in range(self.N):
                if abs(k_idx - l_idx) <= self.L:
                    non_linear_term[k_idx] += a[l_idx] * a[k_idx-l_idx+self.L] * (k_idx - l_idx) * self.w[k_idx-l_idx+self.L]
            non_linear_term[k_idx] *= 4*np.pi**2*(k_idx-self.L) / self.d**2
        return non_linear_term

    def _compute_non_linear_term(self, a):
        """
        Computes sum_j sum_k a_j a_k T_ijk using the structure of T.
        """
        non_linear_term = np.zeros(self.N, dtype=np.complex128)
        for k_idx in range(self.N):
            l_idx_array = np.arange(max(0, k_idx - self.L), min(self.N, k_idx + self.L + 1))
            shift = k_idx - l_idx_array + self.L
            l_idx = l_idx_array
            terms = a[l_idx] * a[shift] * (k_idx - l_idx) * self.w[shift]
            non_linear_term[k_idx] = np.sum(terms)
        non_linear_term *= (np.arange(self.N) - self.L) * 4 * np.pi**2 / self.d**2
        return non_linear_term

    def _compute_K_matrix(self, bar_mu_k=None):
        """
        Compute the matrix K = <mu_bar(W' * phi_j) + phi_j(W' * mu_bar), phi_i'>
        """
        if bar_mu_k is None:
            bar_mu_k = self.bar_mu_k

        self.K = np.zeros((self.N, self.N), dtype=np.complex128)
        for k_idx in range(self.N):
            l_idx = np.arange(max(0, k_idx - self.L),  min(k_idx + self.L, 2*self.L) + 1)
            self.K[k_idx, l_idx] = 4*np.pi**2 / self.d**2 * (k_idx - self.L) * bar_mu_k[k_idx-l_idx+self.L]
            self.K[k_idx, l_idx] *= (l_idx - self.L) * self.w[l_idx] + (k_idx - l_idx) * self.w[k_idx-l_idx+self.L]
        return self.K
    
    def linearized_uncontrolled_solver(self, t_span, t_eval=None):
        """Solve the linearized and uncontrolled McKean-Vlasov equation."""
        def ode_system(t, a):
            return -(self.L_G + self.sigma * self.D + self.K) @ a
        sol = solve_ivp(ode_system, t_span, self.a0, t_eval=t_eval)
        return sol
    
    def nonlinear_uncontrolled_solver_y(self, t_span, t_eval=None):
        """Solve the non-linear and uncontrolled McKean-Vlasov equation."""
        def ode_system(t, a):
            nonlinear_term = self._compute_non_linear_term(a)
            return -(self.L_G + self.sigma * self.D + self.K) @ a - nonlinear_term
        sol = solve_ivp(ode_system, t_span, self.a0, t_eval=t_eval)
        return sol

    def nonlinear_uncontrolled_solver_mu(self, t_span, t_eval=None):
        """Solve the non-linear and uncontrolled McKean-Vlasov equation."""
        def ode_system(t, a):
            nonlinear_term = self._compute_non_linear_term(a)
            return -(self.L_G + self.sigma * self.D) @ a - nonlinear_term
        sol = solve_ivp(ode_system, t_span, self.mu0_projected, t_eval=t_eval)
        return sol

    def nonlinear_controlled_solver_mu(self, t_span, t_eval=None, u=lambda t: np.zeros_like(t)):
        """Solve the non-linear and uncontrolled McKean-Vlasov equation."""
        def ode_system(t, a, u):
            nonlinear_term = self._compute_non_linear_term(a)
            return -(self.L_G + self.sigma * self.D - u(t) * self.Psi) @ a - nonlinear_term
        sol = solve_ivp(ode_system, t_span, self.mu0_projected, t_eval=t_eval, args=(u,))
        return sol

    def nonlinear_controlled_solver_y(self, t_span, t_eval=None, u=lambda t,a: np.zeros_like(t)):
        """Solve the non-linear and controlled McKean-Vlasov equation for y."""
        def ode_system(t, a, u):
            nonlinear_term = self._compute_non_linear_term(a)
            return -(self.L_G + self.sigma * self.D + self.K + u(t,a) * self.Psi) @ a - nonlinear_term - u(t,a) * self.b
        sol = solve_ivp(ode_system, t_span, self.a0, t_eval=t_eval, args=(u,))
        return sol

    def linear_controlled_solver_y(self, t_span, t_eval=None, u=lambda t,a: np.zeros_like(t)):
        """Solve the linear and controlled McKean-Vlasov equation for y."""
        def ode_system(t, a, u):
            return -(self.L_G + self.sigma * self.D + self.K) @ a - u(t,a) * self.b + self.delta * a
        sol = solve_ivp(ode_system, t_span, self.a0, t_eval=t_eval, args=(u,))
        return sol

    def solve_riccati(self):
        """Solve the Riccati equation to compute Pi."""
        A = -self.L_G - self.sigma * self.D - self.K + self.delta * np.eye(self.L_G.shape[0])
        self.B = -np.reshape(self.b, (-1,1))
        
        self.check_are_conditions(A, self.B, self.M, np.eye(self.B.shape[1]))

        self.Pi = solve_continuous_are(
            A, self.B, self.M, np.eye(self.B.shape[1])
        )

    def solve_control_linearized_problem(self, t_span, t_eval=None):
        """Solve the linearized and controlled McKean-Vlasov equation."""
        if self.Pi == None:
            self.solve_riccati()
        sol = self.linear_controlled_solver_y(t_span, t_eval, u=lambda t,a: -self.B.conj().T @ self.Pi @ a)
        return sol

    def solve_control_problem(self, t_span, t_eval=None):
        """Solve the non-linear and controlled McKean-Vlasov equation."""
        if self.Pi == None:
            self.solve_riccati()
        sol = self.nonlinear_controlled_solver_y(t_span, t_eval, u=lambda t,a: -self.B.conj().T @ self.Pi @ a)
        return sol

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
        unstable_eigenvalues = [ev for ev in eigvals(A) if np.real(ev) >= 0]
        for eigenvalue in unstable_eigenvalues:
            extended_matrix = np.hstack(
                [eigenvalue * np.eye(A.shape[0]) - A, B]
            )
            rank_C = np.linalg.matrix_rank(extended_matrix)
            assert rank_C == A.shape[0], \
                    "The pair (A, B) is not stabilizable, rank(C) = {} < rank(A) = {}, eigenvalues = {}".format(rank_C, A.shape[0], unstable_eigenvalues)
        print("All conditions satisfied. Matrices are suitable for solving the ARE.")

class TestNonLinearTerm(unittest.TestCase):
    
    def setUp(self):

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

    def test_performance_comparison(self):
        # Measure performance of the original function
        start_time_original = time.time()
        result_original = self.model.compute_bar_mu_method2()
        time_original = time.time() - start_time_original

        # Measure performance of the improved function
        start_time_improved = time.time()
        result_improved = self.model.compute_bar_mu_method3()
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
        result_original = self.model.compute_bar_mu_method2()
        result_improved = self.model.compute_bar_mu_method3()

        # Use assertArrayAlmostEqual from numpy.testing to compare arrays
        np.testing.assert_array_almost_equal(result_original, result_improved, decimal=6,
                                             err_msg="The outputs of the original and improved functions do not match.")

if __name__ == '__main__':

    def G(x):
        return (x - np.pi)**2

    def alpha(x):
        return np.ones_like(x)

    def W(x):
        return (x - np.pi)**2

    def mu_0(x):
        alpha_param = 2.0
        beta_param = 2.0
        Z = (2 * np.pi)**(alpha_param + beta_param - 1) * beta(alpha_param, beta_param)
        return (x**(alpha_param - 1) * (2 * np.pi - x)**(beta_param - 1)) / Z
    
    #unittest.main()

    # Initialize the solver
    solver = McKeanVlasovSolver(L=30, d=2*np.pi, G=G, alpha=alpha, W=W, mu_0=mu_0, min_fourier_samples=2000, delta=0.0)
    
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)
    x = np.linspace(0, 2*np.pi, num=1000)

    solution = solver.solve_control_problem(t_span, t_eval)
    control = [-solver.B.conj().T @ solver.Pi @ solution.y[:, i] for i in range(len(solution.t))]

    #solution1 = solver.linearized_uncontrolled_solver(t_span, t_eval)
    #solution2 = solver.nonlinear_uncontrolled_solver_y(t_span, t_eval)

    #time_points = solution1.t
    #coefficients1 = solution1.y
    #coefficients2 = solution2.y

    #vect = [np.sum(abs(solver.reconstruction(coefficients1[:,i], x) - solver.reconstruction(coefficients2[:,i], x))) for i in range(len(time_points))]
    
    #plt.plot(time_points, vect, label="Initial")
    #plt.show()
    #plt.plot(x, , label="Final")

    #time_points = solution.t
    #coefficients = solution.y
    #vect = solver.reconstruction(coefficients[:,-1], x)

    #plt.plot(x, vect)
    #plt.show()
    #plt.plot(time_points, control)
    #plt.legend()
    #plt.show()