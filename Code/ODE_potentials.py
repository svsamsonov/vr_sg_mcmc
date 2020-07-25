import numpy as np
from scipy.integrate import RK45,solve_ivp

class VanDerPolePotential:
    """
    class for inference on Van-der-Pole oscillator equations
    """
    def __init__(self,sigma,sigma_prior,t_moments,theta,y0,t0,t_bound):
        self.theta = theta
        self.true_theta = theta
        self.sigma = sigma
        self.t_moments = t_moments
        self.y0 = y0
        self.y0_embed = np.array([0.0,0.0],dtype=float)
        #initial time
        self.t0 = t0
        #last time
        self.t_bound = t_bound
        #default solver - Runge-Kutta of order 5(4)
        self.solver_type = 'RK45'
        self.syst_solver = None
        self.embed_syst_solver = None
        self.X = np.zeros(len(self.t_moments),dtype=float)
        self.Embed = np.zeros(len(self.t_moments),dtype=float)
        self.rtol = 1e-3
        self.atol = 1e-5
        self.sigma_prior = sigma_prior
        self.Y = self.init_y(sigma)
        
    def init_y(self,sigma):
        """
        function to solve ODE system for the first time, returns observations, corrupted by normal noise
        """
        np.random.seed(1821)
        solution = solve_ivp(self.van_der_Pole,t_span=(self.t0,self.t_bound),y0=self.y0,method='RK45',dense_output=True, vectorized=False,rtol = self.rtol, atol = self.atol)
        self.syst_solver = solution.sol
        #update embedding system solver
        embed_solution = solve_ivp(self.embed_van_der_Pole,t_span=(self.t0,self.t_bound),y0=self.y0_embed,method='RK45',dense_output=True, vectorized=False,rtol=self.rtol,atol=self.atol)
        self.embed_syst_solver = embed_solution.sol
        #create vector of observations, corrupt by noise
        Y = np.zeros(len(self.t_moments),dtype=float)
        for i in range(len(self.t_moments)):
            Y[i] = self.syst_solver(self.t_moments[i])[0] + self.sigma*np.random.randn()
        return Y
        
    def van_der_Pole(self,t,x):
        """
        right-hand side for van-der-Pole system of ODE;
        Args:
            t - scalar, 
            x - np.array of shape (2,) - trajectory
        Returns:
            y - np.array of shape (2,)
        """
        y = np.zeros_like(x)
        y[0] = x[1]
        y[1] = self.theta*(1-x[0]**2)*x[1] - x[0]
        return y

    def embed_van_der_Pole(self,t,u):
        """
        embedding equations for van-der-Pole system of ODE;
        Args:
            t - scalar,
            u - np.array of shape (2,) - embdedding variables
        Returns:
            y - np.array of shape (2,)
        """
        y = np.zeros_like(u)
        x = self.syst_solver(t)
        y[0] = u[1]
        y[1] = (1-x[0]**2)*x[1] - 2*self.theta*u[0]*x[0]*x[1] + self.theta*(1-x[0]**2)*u[1] - u[0]
        return y
    
    def update_theta(self,theta):
        self.theta = theta
        return
    
    def update_system_solvers(self):
        """
        Solve initial system and embedding systems, put new solvers to their addresses
        """
        #update system solver
        solution = solve_ivp(self.van_der_Pole,t_span=(self.t0,self.t_bound),y0=self.y0,method='RK45',dense_output=True, vectorized=False,rtol=self.rtol,atol=self.atol)
        self.syst_solver = solution.sol
        #update embedding system solver
        embed_solution = solve_ivp(self.embed_van_der_Pole,t_span=(self.t0,self.t_bound),y0=self.y0_embed,method='RK45',dense_output=True, vectorized=False,rtol=self.rtol,atol=self.atol)
        self.embed_syst_solver = embed_solution.sol
        for i in range(len(self.t_moments)):
            self.X[i] = self.syst_solver(self.t_moments[i])[0]
            self.Embed[i] = self.embed_syst_solver(self.t_moments[i])[0]
        return
    
    #def log_prior(self,theta):
        """
        evaluates log-normal prior at point theta
        """
        #return -0.5*np.log(2*np.pi) - np.log(self.sigma_prior) -np.log(theta)- 1./(2*self.sigma_prior**2)*np.log(theta)**2
    
    def log_prior(self,theta):
        """
        evaluates normal prior at point theta
        """
        return - 1./(2*self.sigma_prior**2)*(theta-self.true_theta)**2
        
    
    #def grad_log_prior(self,theta):
        """
        evaluates gradient of log-normal prior at point theta
        """
        #return  -1./theta - np.log(theta)/(theta*self.sigma_prior**2)
        
    def grad_log_prior(self,theta):
        """
        evaluates gradient of normal prior at point theta
        """
        return - 1./(self.sigma_prior**2)*(theta-self.true_theta)
    
    def log_likelihood(self,theta):
        """
        evaluates log-likelihood at point theta
        """
        return (-0.5*np.log(2*np.pi) - np.log(self.sigma))*len(self.Y) - (1./(2*self.sigma**2))*np.sum((self.Y-self.X)**2)
    
    def grad_log_likelihood(self,theta):
        """
        evaluates gradient of log-likelihood at point theta 
        """
        return (1./self.sigma**2)*np.dot(self.Y-self.X,self.Embed)
        
    def log_potential(self,theta,t=1.0):
        """
        evaluates logarithm of power posterior, with t \in [0,1]
        """
        return self.log_prior(theta) + t*self.log_likelihood(theta)
    
    def grad_log_potential(self,theta,t=1.0):
        """
        evaluates gradient of logarithm of power posterior, with t \in [0,1]
        """
        return self.grad_log_prior(theta) + t*self.grad_log_likelihood(theta)
    
    
class LotkiVolterraPotential:
    """
    class for inference on Lotki-Volterra dynamical system
    """
    def __init__(self,sigma,mu_prior,sigma_prior,t_moments,theta,y0,t0,t_bound):
        """
        Args:
            theta - np.array of shape (4,) - respectively, (alpha,beta,gamma,delta) parameters of dynamical system;
            sigma - np.array of shape (2,) - standard deviations for errors in victims and predators populations respectively;
            sigma_prior - np.array of shape (4,) - standard deviations for priors; 
        """
        self.theta = theta
        self.theta_mle = np.zeros_like(theta)
        self.sigma = sigma
        self.t_moments = t_moments
        self.y0 = y0
        self.y0_embed = np.zeros(8,dtype=float)
        #initial time
        self.t0 = t0
        #last time
        self.t_bound = t_bound
        #default solver - Runge-Kutta of order 5(4)
        self.solver_type = 'RK45'
        self.syst_solver = None
        self.embed_syst_solver = None
        self.X = np.zeros((len(self.t_moments),2),dtype=float)
        self.Embed = np.zeros((len(self.t_moments),8),dtype=float)
        self.rtol = 1e-3
        self.atol = 1e-3
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.Y = self.init_y(sigma)
        #add inside the logarithm for numerical stability
        self.eps = 1e-2
        
    def init_y(self,sigma):
        """
        function to solve ODE system for the first time, returns observations, corrupted by normal noise
        """
        np.random.seed(1821)
        solution = solve_ivp(self.lotki_volterra,t_span=(self.t0,self.t_bound),y0=self.y0,method='RK45',dense_output=True, vectorized=False,rtol = self.rtol, atol = self.atol)
        self.syst_solver = solution.sol
        print("system solved")
        #update embedding system solver
        embed_solution = solve_ivp(self.embed_lotki_volterra,t_span=(self.t0,self.t_bound),y0=self.y0_embed,method='RK45',dense_output=True, vectorized=False,rtol=self.rtol,atol=self.atol)
        self.embed_syst_solver = embed_solution.sol
        #create vector of observations, corrupt by noise
        Y = np.zeros((len(self.t_moments),2),dtype=float)
        for i in range(len(self.t_moments)):
            #Y[i] = self.syst_solver(self.t_moments[i]) + self.sigma*np.random.randn(2)
            Y[i] = np.exp(np.log(self.syst_solver(self.t_moments[i])) + self.sigma*np.random.randn(2))
        print(Y)
        return Y
        
    def lotki_volterra(self,t,x):
        """
        right-hand side for Lotki-Volterra systen of ODE's (predator-victim moder);
        Args:
            t - scalar,
            x - np.array of shape (2,) - trajectory
        Returns:
            y - np.array of shape (2,)
        """
        alpha = self.theta[0]
        beta = self.theta[1]
        gamma = self.theta[2]
        delta = self.theta[3]
        y = np.zeros_like(x)
        y[0] = (alpha - beta*x[1])*x[0]
        y[1] = (-gamma + delta*x[0])*x[1]
        return y

    def embed_lotki_volterra(self,t,u):
        """
        embedding equations for van-der-Pole system of ODE;
        Args:
            t - scalar,
            u - np.array of shape (8,) - embdedding variables
        Returns:
            y - np.array of shape (8,)
        """
        alpha = self.theta[0]
        beta = self.theta[1]
        gamma = self.theta[2]
        delta = self.theta[3]
        y = np.zeros_like(u)
        x = self.syst_solver(t)
        #embedding equations w.r.t. alpha
        y[0] = (1-beta*u[1])*x[0] + (alpha-beta*x[1])*u[0]
        y[1] = delta*u[0]*x[1] + (-gamma+delta*x[0])*u[1]
        #embedding equations w.r.t. beta
        y[2] = (-x[1]-beta*u[3])*x[0] + (alpha - beta*x[1])*u[2]
        y[3] = delta*u[2]*x[1] + (-gamma + delta*x[0])*u[3]
        #embedding equations w.r.t. gamma
        y[4] = -beta*u[5]*x[0] + (alpha-beta*x[1])*u[4]
        y[5] = (-1+delta*u[4])*x[1] + (-gamma+delta*x[0])*u[5]
        #embedding equations w.r.t. delta
        y[6] = -beta*u[7]*x[0] + (alpha - beta*x[1])*u[6]
        y[7] = (x[0]+delta*u[6])*x[1] + (-gamma+delta*x[0])*u[7]
        return y
    
    def update_theta(self,theta):
        self.theta = theta
        return
    
    def set_mle(self,theta):
        self.theta_mle = theta
        return
    
    def update_system_solvers(self):
        """
        Solve initial system and embedding systems, put new solvers to their addresses
        """
        #update system solver
        solution = solve_ivp(self.lotki_volterra,t_span=(self.t0,self.t_bound),y0=self.y0,method='RK45',dense_output=True, vectorized=False,rtol=self.rtol,atol=self.atol)
        self.syst_solver = solution.sol
        #update embedding system solver
        embed_solution = solve_ivp(self.embed_lotki_volterra,t_span=(self.t0,self.t_bound),y0=self.y0_embed,method='RK45',dense_output=True, vectorized=False,rtol=self.rtol,atol=self.atol)
        self.embed_syst_solver = embed_solution.sol
        for i in range(len(self.t_moments)):
            self.X[i] = self.syst_solver(self.t_moments[i])
            self.Embed[i] = self.embed_syst_solver(self.t_moments[i])
        #print(self.X)
        return
    
    def log_prior(self,theta):
        """
        evaluates normal prior at point theta
        """
        return np.sum((-(theta-self.mu_prior)**2)/(2*self.sigma_prior**2))
    
    def grad_log_prior(self,theta):
        """
        evaluates gradient of gamma prior at point theta
        """
        return -(theta-self.mu_prior)/(self.sigma_prior**2)
    
    def log_likelihood(self,theta):
        """
        evaluates log-likelihood at point theta
        """
        cur_ll = 0.0
        #each parameter has its own standard error, errors are assumed to be independent
        #for i in range(len(self.sigma)):
            #cur_ll += -(1./(2*self.sigma[i]**2))*np.sum((self.Y[:,i]-self.X[:,i])**2)
        for i in range(len(self.sigma)):
            cur_ll -= np.sum(np.log(2*self.sigma[i]*self.Y[:,i]))
            cur_ll -= np.sum(((np.log(self.Y[:,i] + self.eps) - np.log(self.X[:,i] + self.eps))**2)/(2*self.sigma[i]**2))
        return cur_ll
    
    def grad_log_likelihood(self,theta):
        """
        evaluates gradient of log-likelihood at point theta 
        """
        cur_grad = np.zeros_like(self.theta)
        #for i in range(len(self.sigma)):
            #cur_grad += (1./self.sigma[i]**2)*np.sum(self.Embed[:,i::2]*(self.Y[:,i]-self.X[:,i])[:,np.newaxis],axis=0)
        for i in range(len(self.sigma)):
            cur_grad -= (1./self.sigma[i]**2)*np.sum(self.Embed[:,i::2]*\
                  ((np.log(self.X[:,i]+self.eps) - np.log(self.Y[:,i]+self.eps))/self.X[:,i])[:,np.newaxis],axis=0)
        return cur_grad
        
    def log_potential(self,theta,t=1.0):
        """
        evaluates logarithm of power posterior, with t \in [0,1]
        """
        return self.log_prior(theta) + t*self.log_likelihood(theta)
    
    def grad_log_potential(self,theta,t=1.0):
        """
        evaluates gradient of logarithm of power posterior, with t \in [0,1]
        """
        return self.grad_log_prior(theta) + t*self.grad_log_likelihood(theta)

