import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.optimize as opt
from multiprocessing import Pool
import multiprocessing
import copy

def prepare_data(data_train,data_test,size="small"):
    """dataloader to work with movielens100k, it is expected that you have it in the same folder
    Args:
        data_train,data_test - pandas dataframes with loaded movilens baselines;
        size - "small" or "full"
    returns:
        splitted into train/test parts datasets, loaded into numpy arrays with rating converted into float
    """
    if size == "full":
        Train = copy.deepcopy(data_train.iloc[:,:-1])
        Test = copy.deepcopy(data_test.iloc[:,:-1])
        return np.asarray(Train),np.asarray(Test)
    elif size == "small":
        D = 101
        Train = copy.deepcopy(data_train[data_train.iloc[:,0] < D].iloc[:,:-1])
        Test = copy.deepcopy(data_test[data_test.iloc[:,0] < D].iloc[:,:-1])
        return np.asarray(Train),np.asarray(Test)
    else:
        raise "Not implemented error in prepare_data: incorrect parameter size"

class bayes_PCA_potential:
    """ implementing a potential U = logarithm of the posterior distribution
        given for online bayesian matrix factorization
    """
    def __init__(self,D,Train,Test,tau,batch_size,N_users,N_films,r_seed):
        """ initialization 
        Args:
        """
        #D - dimenstion of the latent space
        self.D = D
        self.N_users = N_users
        self.N_films = N_films
        np.random.seed(r_seed)
        #generate lambdas from Gamma(1,1) distribution
        self.lambda_u = np.random.exponential(scale=1.0)
        self.lambda_v = np.random.exponential(scale=1.0)
        self.lambda_a = np.random.exponential(scale=1.0)
        self.lambda_b = np.random.exponential(scale=1.0)
        #fix variance tau 
        self.tau = tau
        #train and test samples
        self.Train = Train
        self.Test = Test
        self.batch_size = batch_size
        #current parameters U,V,a,b
        self.U = np.random.randn(self.D,self.N_users)/np.sqrt(self.lambda_u)
        self.V = np.random.randn(self.D,self.N_films)/np.sqrt(self.lambda_v)
        self.a = np.random.randn(self.N_users)/np.sqrt(self.lambda_a)
        self.b = np.random.randn(self.N_films)/np.sqrt(self.lambda_b)
        #Star parameters for 
        self.U_star = np.zeros_like(self.U)
        self.V_star = np.zeros_like(self.V)
        self.a_star = np.zeros_like(self.a)
        self.b_star = np.zeros_like(self.b)
        #current gradients w.r.t. parameters, initialize randomly
        self.grad_U = np.zeros_like(self.U)
        self.grad_V = np.zeros_like(self.V)
        self.grad_a = np.zeros_like(self.a)
        self.grad_b = np.zeros_like(self.b)
        #dimensions
        self.traj_grad_SAGA = None
        self.total_dim = self.grad_U.shape[0]*self.grad_U.shape[1] + self.grad_V.shape[0]*self.grad_V.shape[1]+self.grad_U.shape[1]+\
                    self.grad_V.shape[1]
        print("dimension = ",self.total_dim)
    
    def zero_grads(self):
        """zeros gradients w.r.t. U,V,a,b
        """
        self.grad_U.fill(0.0)
        self.grad_V.fill(0.0)
        self.grad_a.fill(0.0)
        self.grad_b.fill(0.0)
        return  
    
    def concat_grad(self):
        """Puts together gradients w.r.t. U,V,a,b into one 1-dimensional vector
        """
        dim = self.dim_U+self.dim_V+len(self.grad_a)+len(self.grad_b)
        print("dimensionality = ",dim)        
        full_grad = np.zeros(dim,dtype=float)
        full_grad[:self.dim_U] = self.grad_U.ravel()
        full_grad[self.dim_U:self.dim_U+self.dim_V] = self.grad_V.ravel()
        full_grad[self.dim_U+self.dim_V:self.dim_U+self.dim_V+len(self.grad_a)] = self.grad_a.ravel()
        full_grad[self.dim_U+self.dim_V+len(self.grad_a):] = self.grad_b.ravel()
        return full_grad
    
    def compute_full_grad(self):
        self.zero_grads()
        for i in range(len(self.X)):
            elem = i
            usr_ind = self.X[i][0]
            film_ind = self.X[i][1]
            lin_part = self.Y[elem]-np.dot(self.U_star[:,usr_ind],self.V_star[:,film_ind])-self.a_star[usr_ind]-\
                        self.b_star[film_ind]
            #update gradients
            self.grad_U[:,usr_ind] += self.tau*self.V_star[:,film_ind]*lin_part
            self.grad_V[:,film_ind] += self.tau*self.U_star[:,usr_ind]*lin_part
            self.grad_a[usr_ind] += self.tau*lin_part
            self.grad_b[film_ind] += self.tau*lin_part
        #account for priors now
        self.grad_U -= self.lambda_u*self.U_star
        self.grad_V -= self.lambda_v*self.V_star
        self.grad_a -= self.lambda_a*self.a_star
        self.grad_b -= self.lambda_b*self.b_star 
        print("grad_U norm:",np.linalg.norm(self.grad_U[:,:3]))
        print("U norm:",np.linalg.norm(self.U_star[:,:3]))
        print("grad_V norm: ",np.linalg.norm(self.grad_V[:,:3]))
        print("grad_a norm:",np.linalg.norm(self.grad_a))
        print("grad_b norm",np.linalg.norm(self.grad_b))
        return self.concat_grad()
        
    def SGD(self,N_steps_max,gamma_start,decay_rate,batch_size,alpha):
        """Implements stochastic gradient descent w.r.t. parameters U,V,a,b with fixed lambdas
        """
        #initialize array of step sizes
        np.random.seed(777)
        gammas = gamma_start*(1+np.arange(N_steps_max)/decay_rate)**(alpha)
        self.init_theta_star()
        ratio = len(self.X)/batch_size
        print("ratio = ",ratio)
        scale_U = 0.0
        scale_V = 0.0
        scale_a = 0.0
        scale_b = 0.0
        for k in range(N_steps_max):
            cur_gamma = gammas[k]
            batch_inds = np.random.choice(self.p,batch_size,replace=True)
            #update gradients
            self.zero_grads()
            for i in range(len(batch_inds)):
                elem = batch_inds[i]
                usr_ind = self.X[elem,0]
                film_ind = self.X[elem,1]
                lin_part = self.Y[elem]-np.dot(self.prev_U[:,usr_ind],self.prev_V[:,film_ind])-self.prev_a[usr_ind]-\
                            self.prev_b[film_ind]
                #update gradients
                self.grad_U[:,usr_ind] += self.tau*self.prev_V[:,film_ind]*lin_part
                self.grad_V[:,film_ind] += self.tau*self.prev_U[:,usr_ind]*lin_part
                self.grad_a[usr_ind] += self.tau*lin_part
                self.grad_b[film_ind] += self.tau*lin_part
            self.grad_U*=ratio
            self.grad_V*=ratio
            self.grad_a*=ratio
            self.grad_b*=ratio
            #account for priors now
            self.grad_U -= self.lambda_u*self.prev_U
            self.grad_V -= self.lambda_v*self.prev_V
            self.grad_a -= self.lambda_a*self.prev_a
            self.grad_b -= self.lambda_b*self.prev_b
            #calculate step sizes
            #scale_U_new = np.abs(1./(k+1)*(1./self.dim_U*np.linalg.norm(self.grad_U)**2 + k*scale_U))
            #scale_V_new = np.abs(1./(k+1)*(1./self.dim_V*np.linalg.norm(self.grad_V)**2 + k*scale_V))
            #scale_a_new = np.abs(1./(k+1)*(1./len(self.grad_a)*np.linalg.norm(self.grad_a)**2 + k*scale_a))
            #scale_b_new = np.abs(1./(k+1)*(1./len(self.grad_b)*np.linalg.norm(self.grad_b)**2 + k*scale_b))
            #update old scales
            #scale_U = scale_U_new
            #scale_V = scale_V_new
            #scale_a = scale_a_new
            #scale_b = scale_b_new
            #update parameters now
            #self.U_star += gamma_start*self.grad_U#/np.sqrt(scale_U_new)
            #self.V_star += gamma_start*self.grad_V#/np.sqrt(scale_V_new)
            #self.a_star += gamma_start*self.grad_a#/np.sqrt(scale_a_new)
            #self.b_star += gamma_start*self.grad_b/np.sqrt(scale_b_new) 
            if (k % decay_rate == 0):
                print("full gradient norm = ",np.linalg.norm(self.compute_full_grad()))
        return       
        
    def compute_MAP_determ(self,print_info,optim_params):
        """Compute MAP estimation either by stochastic gradient or by deterministic gradient descent
        """
        #by default
        if optim_params["compute_fp"] == False:
            return np.zeros(self.d, dtype = np.float64)
        n_restarts = optim_params["n_restarts"]
        tol = optim_params["gtol"]
        sigma = optim_params["sigma"]
        order = optim_params["order"]
        converged = False
        cur_f = 1e100
        cur_x = np.zeros(self.d,dtype = np.float64)
        best_jac = None
        for n_iters in range(n_restarts):
            if order == 2:#Newton-CG, 2nd order
                vspom = opt.minimize(self.minus_potential,sigma*np.random.randn(self.d),method = "Newton-CG", jac = self.gradpotential_deterministic, hess = self.hess_potential_determ, tol=tol)
            elif order == 1:#BFGS, quasi-Newtion, almost 1st order
                vspom = opt.minimize(self.minus_potential,sigma*np.random.randn(self.d), jac = self.gradpotential_deterministic, tol=tol)
            else:
                raise "not implemented error: order of optimization method should be 1 or 2"
            converged = converged or vspom.success
            if print_info:#show some useless information
                print("optimization iteration ended")
                print("success = ",vspom.success)
                print("func value = ",vspom.fun)
                print("jacobian value = ",vspom.jac)
                print("number of function evaluation = ",vspom.nfev)
                print("number of jacobian evaluation = ",vspom.njev)
                print("number of optimizer steps = ",vspom.nit)
            if vspom.fun < cur_f:
                cur_f = vspom.fun
                cur_x = vspom.x
                best_jac = vspom.jac
        theta_star = cur_x
        if converged:
            print("theta^* found succesfully")
        else:
            print("requested precision not necesserily achieved during searching for theta^*, try to increase error tolerance")
        print("final jacobian at termination: ")
        print(best_jac)
        return theta_star
    
    def grad_descent(self,rand_seed,print_info,optim_params,typ):
        """repeats gradient descent until convergence for one starting point
        """
        stochastic = optim_params["stochastic"]
        batch_size = optim_params["batch_size"]
        sigma = optim_params["sigma"]
        gtol = optim_params["gtol"]
        gamma = optim_params["gamma"]
        weight_decay = optim_params["weight_decay"]
        N_iters = optim_params["loop_length"]
        Max_loops = optim_params["n_loops"]
        cur_jac_norm = 1e100
        loops_counter = 0
        np.random.seed(rand_seed)
        x = sigma*np.random.randn(self.d)
        while ((cur_jac_norm > gtol) and (loops_counter < Max_loops)):
            #if true gradient still didn't converge => do SGD
            print("jacobian norm = %f, step size = %f, loop number = %d" % (cur_jac_norm,gamma,loops_counter))
            for i in range(N_iters):
                if stochastic:#SGD
                    batch_inds = np.random.choice(self.p,batch_size)
                    grad = (self.p/batch_size)*self.gradloglikelihood_stochastic(x,batch_inds) + self.gradlogprior(x)
                else:#deterministic GD
                    grad = self.gradloglikelihood_determ(x) + self.gradlogprior(x)
                x = x + gamma*grad
            gamma = gamma*weight_decay
            cur_jac_norm = np.linalg.norm(self.gradpotential_deterministic(x))
            loops_counter += 1
        res = {"value":self.minus_potential(x),"x":x,"jac_norm":cur_jac_norm}
        return res
    
    def compute_MAP_gd(self,print_info,optim_params):
        """Compute MAP estimation by stochastic gradient ascent
        """
        if optim_params["compute_fp"] == False:
            return np.zeros(self.d, dtype = np.float64)
        cur_f = 1e100
        cur_x = np.zeros(self.d,dtype = np.float64)
        best_jac = None
        n_restarts = optim_params["n_restarts"]
        nbcores = multiprocessing.cpu_count()
        trav = Pool(nbcores)
        res = trav.starmap(self.grad_descent, [(777+i,print_info,optim_params) for i in range (n_restarts)])
        for ind in range(len(res)):
            if res[ind]["value"] < cur_f:
                cur_f = res[ind]["value"]
                cur_x = res[ind]["x"]
                best_jac = res[ind]["jac_norm"]
        theta_star = cur_x
        print("best jacobian norm = ",best_jac)
        return theta_star
    
    def hess_potential_determ(self,theta):
        """Second-order optimization to accelerate optimization and (possibly) increase precision
        """
        XTheta = self.X @ theta
        term_exp = np.divide(np.exp(-XTheta/2),1 + np.exp(XTheta))
        X_add = self.X*term_exp.reshape((self.p,1))
        #second summand comes from prior
        return np.dot(X_add.T,X_add) + 1./self.varTheta
        
    def loglikelihood(self,theta):
        """loglikelihood of the Bayesian regression
        Args:
            theta: parameter of the state space R^d where the likelihood is
                evaluated
        Returns:
            real value of the likelihood evaluated at theta
        """
        if self.type == "g": # Linear regression
            return -(1. / (2*self.varY))* np.linalg.norm(self.Y-np.dot(self.X,theta))**2 \
                        - (self.d/2.)*np.log(2*np.pi*self.varY)
        elif self.type == "l": # Logistic
            XTheta = np.dot(-self.X, theta)
            temp1 = np.dot(1.0-self.Y, XTheta)
            temp2 = -np.sum(np.log(1+np.exp(XTheta)))
            return temp1+temp2
        else: # Probit
            cdfXTheta = spstats.norm.cdf(np.dot(self.X, theta))
            cdfMXTheta = spstats.norm.cdf(-np.dot(self.X, theta))
            temp1 = np.dot(self.Y, np.log(cdfXTheta))
            temp2 = np.dot((1 - self.Y), np.log(cdfMXTheta))
            return temp1+temp2
    
    def gradloglikelihood_determ(self,theta):
        """Purely deterministic gradient of log-likelihood, used for theta^* search
        Returns:
            R^d vector of the (full and fair) gradient of log-likelihood, evaluated at theta^*
        """
        if self.type == "g": # Linear
            temp1 = np.dot(np.dot(np.transpose(self.X), self.X), theta)
            temp2 = np.dot(np.transpose(self.X), self.Y)
            return (1. / self.varY)*(temp2 - temp1)
        elif self.type == "l": # Logistic
            temp1 = np.exp(np.dot(-self.X, theta))
            temp2 = np.dot(np.transpose(self.X), self.Y)
            temp3 = np.dot(np.transpose(self.X), np.divide(1, 1+temp1))
            return temp2 - temp3
        else: # Probit
            XTheta = np.dot(self.X, theta)
            logcdfXTheta = np.log(spstats.norm.cdf(XTheta))
            logcdfMXTheta = np.log(spstats.norm.cdf(-XTheta))
            temp1 = np.multiply(self.Y, np.exp(-0.5*(np.square(XTheta)+np.log(2*np.pi)) \
                                               -logcdfXTheta))
            temp2 = np.multiply((1 - self.Y), np.exp(-0.5*(np.square(XTheta)+np.log(2*np.pi)) \
                                               -logcdfMXTheta))
            return np.dot(np.transpose(self.X), temp1-temp2)
        
    def gradloglikelihood_stochastic(self,theta,batch_inds):
        """returns stochastic gradient estimation over batch_inds observations
        Args:
            ...
        Returns:
            ...
        """
        data = self.X[batch_inds,:]
        y_data = self.Y[batch_inds]
        if self.type == "g":#Linear
            raise "Not implemented error in gradloglikelihood stochastic"
        elif self.type == "l":#Logistic
            temp1 = np.exp(-np.dot(data, theta))
            temp2 = np.dot(np.transpose(data), y_data)
            temp3 = np.dot(np.transpose(data), np.divide(1, 1+temp1))
            return temp2 - temp3
        else:#Probit
            raise "Not implemented error in gradloglikelihood stochastic"
    
    def evaluate(self,test):
        """Evaluates MSE based on current estimates U,V,a,b;
            if test = True - on test dataset;
            if train = False - on train dataset;
        returns: MSE
        """
        if test:#calculate error on test data
            mse_err = (self.Test[:,2] - inner1d(self.U[:,self.Test[:,0]].T,self.V[:,self.Test[:,1]].T) - self.a[self.Test[:,0]]-\
                self.b[self.Test[:,1]])**2
        else:#calculate error on training data
            mse_err = np.zeros(len(self.Train),dtype = float)
            mse_err = (self.Train[:,2] - inner1d(self.U[:,self.Train[:,0]].T,self.V[:,self.Train[:,1]].T) - self.a[self.Train[:,0]]-\
                self.b[self.Train[:,1]])**2
        return np.mean(mse_err)
        
    
    def SAGA_step(self):
        """
        """
        ratio = len(self.X_train)/self.batch_size
        batch_inds = np.random.choice(len(self.X_train),self.batch_size,replace=True)
        #update gradients
        grad_U_new = np.zeros_like(self.U)
        grad_V_new = np.zeros_like(self.V)
        grad_a_new = np.zeros_like(self.a)
        grad_b_new = np.zeros_like(self.b)
        #copy current gradients
        grad_U_old = copy.deepcopy(self.grad_U)
        grad_V_old = copy.deepcopy(self.grad_V)
        grad_a_old = copy.deepcopy(self.grad_a)
        grad_b_old = copy.deepcopy(self.grad_b)
        #compute SAGA updates
        lin_part = self.Train[batch_inds,2] - inner1d(self.U[:,self.Train[batch_inds,0]].T,self.V[:,self.Train[batch_inds,1]].T)-\
            self.a[self.Train[batch_inds,0]] - self.b[self.Train[batch_inds,1]]
        grad_U_new[:,self.Train[batch_inds,0]] += self.tau*self.V[:,self.Train[batch_inds,1]]*lin_part - self.traj_grad_SAGA[self.Train[batch_inds,0],:self.D]
        grad_V_new[:,self.Train[batch_inds,1]] += self.tau*self.U[:,self.Train[batch_inds,0]]*lin_part - self.traj_grad_SAGA[self.Train[batch_inds,1],self.D:2*self.D]
        grad_a_new[self.Train[batch_inds,0]] += self.tau*lin_part - self.traj_grad_SAGA[self.Train[batch_inds,0],-2]
        grad_b_new[self.Train[batch_inds,1]] += self.tau*lin_part - self.traj_grad_SAGA[self.Train[batch_inds,1],-2]
        #update SAGA estimates
        grad_U = self.grad_U + ratio*grad_U_new
        grad_V = self.grad_V + ratio*grad_V_new
        grad_a = self.grad_a + ratio*grad_a_new
        grad_b = self.grad_b + ratio*grad_b_new
        #account for priors now
        grad_U -= self.lambda_u*self.U
        grad_V -= self.lambda_v*self.V
        grad_a -= self.lambda_a*self.a
        grad_b -= self.lambda_b*self.b
        #update new estimates
        self.grad_U += grad_U_new
        self.grad_V += grad_V_new
        self.grad_a += grad_a_new
        self.grad_b += grad_b_new
        #update cumulative part now
        self.traj_grad_SAGA[self.Train[batch_inds,0],:self.D] = self.tau*self.V[:,self.Train[batch_inds,1]]*lin_part
        self.traj_grad_SAGA[self.Train[batch_inds,1],self.D:2*self.D] = self.tau*self.U[:,self.Train[batch_inds,0]]*lin_part
        self.traj_grad_SAGA[self.Train[batch_inds,0],-2] = self.tau*lin_part
        self.traj_grad_SAGA[self.Train[batch_inds,1],-1] = self.tau*lin_part
        #return previously computed unbiased estimate
        return grad_U,grad_V,grad_a,grad_b
    
    def compute_grads(self, FP = False):
        """Calculate gradients w.r.t. parameters U,V,a,b 
        returns:
            gradient estimates w.r.t U,V,a,b
        """
        ratio = len(self.Train)/self.batch_size
        batch_inds = np.random.choice(len(self.Train),self.batch_size,replace=True)
        #update gradients
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_a = np.zeros_like(self.a)
        grad_b = np.zeros_like(self.b)
        if FP:#optimization, work with star
            cur_U = self.U_star
            cur_V = self.V_star
            cur_a = self.a_star
            cur_b = self.b_star
        else:
            cur_U = self.U
            cur_V = self.V
            cur_a = self.a
            cur_b = self.b
        lin_part = self.Train[batch_inds,2] - inner1d(cur_U[:,self.Train[batch_inds,0]].T,cur_V[:,self.Train[batch_inds,1]].T)-\
            cur_a[self.Train[batch_inds,0]] - cur_b[self.Train[batch_inds,1]]
        grad_U[:,self.Train[batch_inds,0]] += self.tau*cur_V[:,self.Train[batch_inds,1]]*lin_part
        grad_V[:,self.Train[batch_inds,1]] += self.tau*cur_U[:,self.Train[batch_inds,0]]*lin_part
        grad_a[self.Train[batch_inds,0]] += self.tau*lin_part
        grad_b[self.Train[batch_inds,1]] += self.tau*lin_part
        grad_U*=ratio
        grad_V*=ratio
        grad_a*=ratio
        grad_b*=ratio
        #account for priors now
        grad_U -= self.lambda_u*cur_U
        grad_V -= self.lambda_v*cur_V
        grad_a -= self.lambda_a*cur_a
        grad_b -= self.lambda_b*cur_b
        return grad_U,grad_V,grad_a,grad_b
    
    def compute_grads_fp(self):
        """Calculate gradients w.r.t. parameters U,V,a,b based on fixed-point estimates
        returns:
            gradient estimates w.r.t. U,V,a,b
        """
        ratio = len(self.Train)/self.batch_size
        batch_inds = np.random.choice(len(self.Train),self.batch_size,replace=True)
        #update gradients
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_a = np.zeros_like(self.a)
        grad_b = np.zeros_like(self.b)
        #stochastic gradients at given points
        lin_part = self.Train[batch_inds,2] - inner1d(self.U[:,self.Train[batch_inds,0]].T,self.V[:,self.Train[batch_inds,1]].T)-\
            self.a[self.Train[batch_inds,0]] - self.b[self.Train[batch_inds,1]]
        #stochastic gradients at control points
        lin_part_fp = self.Train[batch_inds,2]-inner1d(self.U_star[:,self.Train[batch_inds,0]].T,self.V_star[:,self.Train[batch_inds,1]].T)-\
            self.a_star[self.Train[batch_inds,0]] - self.b_star[self.Train[batch_inds,1]]
        grad_U[:,self.Train[batch_inds,0]] += self.tau*(self.V[:,self.Train[batch_inds,1]]*lin_part-\
                                              self.V_star[:,self.Train[batch_inds,1]]*lin_part_fp)
        grad_V[:,self.Train[batch_inds,1]] += self.tau*(self.U[:,self.Train[batch_inds,0]]*lin_part-\
                                              self.U_star[:,self.Train[batch_inds,0]]*lin_part_fp)
        grad_a[self.Train[batch_inds,0]] += self.tau*(lin_part - lin_part_fp)
        grad_b[self.Train[batch_inds,1]] += self.tau*(lin_part - lin_part_fp)
        grad_U*=ratio
        grad_V*=ratio
        grad_a*=ratio
        grad_b*=ratio
        #account for priors now
        grad_U -= self.lambda_u*(self.U - self.U_star)
        grad_V -= self.lambda_v*(self.V - self.V_star)
        grad_a -= self.lambda_a*(self.a - self.a_star)
        grad_b -= self.lambda_b*(self.b - self.b_star)
        #add gradients at star point
        grad_U += self.grad_U_star
        grad_V += self.grad_V_star
        grad_a += self.grad_a_star
        grad_b += self.grad_b_star
        return grad_U,grad_V,grad_a,grad_b
    
    def compute_full_grads(self, FP = False):
        """Calculate full gradients w.r.t. parameters U,V,a,b
            grad_SAGA - if True, returns also the list of gradients  
        returns:
            gradients w.r.t. U,V,a,b, computed on full train dataset 
        """
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_a = np.zeros_like(self.a)
        grad_b = np.zeros_like(self.b)
        if FP:#computing for fixed-point
            cur_U = self.U_star
            cur_V = self.V_star
            cur_a = self.a_star
            cur_b = self.b_star
        else:#computing for sgld
            cur_U = self.U
            cur_V = self.V
            cur_a = self.a
            cur_b = self.b
        lin_part = self.Train[:,2] - inner1d(cur_U[:,self.Train[:,0]].T,cur_V[:,self.Train[:,1]].T)-\
            cur_a[self.Train[:,0]] - cur_b[self.Train[:,1]]
        grad_U[:,self.Train[:,0]] += self.tau*cur_V[:,self.Train[:,1]]*lin_part
        grad_V[:,self.Train[:,1]] += self.tau*cur_U[:,self.Train[:,0]]*lin_part
        grad_a[self.Train[:,0]] += self.tau*lin_part
        grad_b[self.Train[:,1]] += self.tau*lin_part
        #account for priors now
        grad_U -= self.lambda_u*cur_U
        grad_V -= self.lambda_v*cur_V
        grad_a -= self.lambda_a*cur_a
        grad_b -= self.lambda_b*cur_b
        if FP:#save results
            self.grad_U_star = grad_U
            self.grad_V_star = grad_V
            self.grad_a_star = grad_a
            self.grad_b_star = grad_b
        return grad_U,grad_V,grad_a,grad_b
            
    def SGLD(self,N_burn,N_steps,step,ctrl_steps,r_seed):
        """computes stochastic gradients w.r.t. parameters, do Langevin steps
        """
        np.random.seed(r_seed) 
        mse_train = []
        mse_test = []
        grad_norm = []
        #U,V,a,b are already initialized
        for k in np.arange(N_burn): # burn-in period
            grad_U,grad_V,grad_a,grad_b = self.compute_grads()
            #update
            self.U = self.U + step*grad_U + np.sqrt(2*step)*np.random.standard_normal(size = self.U.shape)
            self.V = self.V + step*grad_V + np.sqrt(2*step)*np.random.standard_normal(size = self.V.shape)
            self.a = self.a + step*grad_a + np.sqrt(2*step)*np.random.standard_normall(size = self.a.shape)
            self.b = self.b + step*grad_b + np.sqrt(2*step)*np.random.standard_normal(size = self.b.shape)
        #burn-in ended
        traj = np.zeros((N_steps // ctrl_steps,self.total_dim),dtype = float)
        traj_grad = np.zeros((N_steps // ctrl_steps,self.total_dim), dtype = float)
        for k in np.arange(N_steps):
            grad_U,grad_V,grad_a,grad_b = self.compute_grads()
            if k % ctrl_steps == 0:
                traj[k // ctrl_steps,:] = np.concatenate((self.U.ravel(),self.V.ravel(),self.a,self.b))
                traj_grad[k // ctrl_steps,:] = np.concatenate((grad_U.ravel(),grad_V.ravel(),grad_a,grad_b))
            self.U = self.U + step*grad_U + np.sqrt(2*step)*np.random.standard_normal(size = self.U.shape)
            self.V = self.V + step*grad_V + np.sqrt(2*step)*np.random.standard_normal(size = self.V.shape)
            self.a = self.a + step*grad_a + np.sqrt(2*step)*np.random.standard_normal(size = self.a.shape)
            self.b = self.b + step*grad_b + np.sqrt(2*step)*np.random.standard_normal(size = self.b.shape)
            if k % ctrl_steps == 0:
                mse_train.append(self.evaluate(test = False))
                mse_test.append(self.evaluate(test = True))
        return traj,traj_grad,np.asarray(mse_train),np.asarray(mse_test)
    
    def SGLD_FP(self,N_opt,N_steps,step_opt,step,ctrl_steps,r_seed):
        """computes stochastic gradients w.r.t. parameters, do Langevin steps
        """
        np.random.seed(r_seed) 
        mse_train = []
        mse_test = []
        grad_norm = []
        #U,V,a,b are already initialized
        #burn-in ended
        self.U_star = copy.deepcopy(self.U)
        self.V_star = copy.deepcopy(self.V)
        self.a_star = copy.deepcopy(self.a)
        self.b_star = copy.deepcopy(self.b)
        for k in np.arange(N_opt): # optimization period, do stochastic gradient descent
            grad_U,grad_V,grad_a,grad_b = self.compute_grads(FP=True)
            #update
            self.U_star = self.U_star + step_opt*grad_U
            self.V_star = self.V_star + step_opt*grad_V
            self.a_star = self.a_star + step_opt*grad_a
            self.b_star = self.b_star + step_opt*grad_b
        self.compute_full_grads(FP = True)
        traj = np.zeros((N_steps // ctrl_steps,self.total_dim),dtype = float)
        traj_grad = np.zeros((N_steps // ctrl_steps,self.total_dim), dtype = float)
        for k in np.arange(N_steps):
            grad_U,grad_V,grad_a,grad_b = self.compute_grads_fp()
            if k % ctrl_steps == 0:
                traj[k // ctrl_steps,:] = np.concatenate((self.U.ravel(),self.V.ravel(),self.a,self.b))
                traj_grad[k // ctrl_steps,:] = np.concatenate((grad_U.ravel(),grad_V.ravel(),grad_a,grad_b))
            self.U = self.U + step*grad_U + np.sqrt(2*step)*np.random.standard_normal(size = self.U.shape)
            self.V = self.V + step*grad_V + np.sqrt(2*step)*np.random.standard_normal(size = self.V.shape)
            self.a = self.a + step*grad_a + np.sqrt(2*step)*np.random.standard_normal(size = self.a.shape)
            self.b = self.b + step*grad_b + np.sqrt(2*step)*np.random.standard_normal(size = self.b.shape)
            if k % ctrl_steps == 0:
                mse_train.append(self.evaluate(test = False))
                mse_test.append(self.evaluate(test = True))
        return traj,traj_grad,np.asarray(mse_train),np.asarray(mse_test)
    
    def SAGA(self,N_steps,step,ctrl_steps,r_seed):
        """computes stochastic gradients w.r.t. parameters, do SAGA steps while maintaining previous gradient estimates
        """
        np.random.seed(r_seed)
        mse_train = []
        mse_test = []
        #initialize starting estimate
        dim = self.N_users + self.N_films + 2
        SAGA_grads = np.zeros(len(self.X_train),dim)
        #compute full gradient
        self.grad_U,self.grad_V,self.grad_a,self.grad_b = self.compute_full_grads(True)
        for k in np.arange(N_steps):
            #here we calculate SAGA update and update the stored values
            grad_U,grad_V,grad_a,grad_b = self.SAGA_step()
            #generate next values
            self.U = self.U + step*grad_U + np.sqrt(2*step)*np.random.normal(size = self.U.shape)
            self.V = self.V + step*grad_V + np.sqrt(2*step)*np.random.normal(size = self.V.shape)
            self.a = self.a + step*grad_a + np.sqrt(2*step)*np.random.normal(size = self.a.shape)
            self.b = self.b + step*grad_b + np.sqrt(2*step)*np.random.normal(size = self.b.shape)
            #print test information or not
            if k % ctrl_steps == 0:
                mse_train.append(self.evaluate(test = False))
                mse_test.append(self.evaluate(test = True))   
        return np.asarray(mse_train),np.asarray(mse_test)
        
    def update_weights(self,r_seed):
        """update weights (i.e. lambdas) 
        """
        np.random.seed(10000 + r_seed)
        self.lambda_u = np.random.gamma(shape = 1 + self.D*self.N_users/2, scale = 1.0/(1+np.sum(self.U**2)/2))
        self.lambda_v = np.random.gamma(shape = 1 + self.D*self.N_films/2, scale = 1.0/(1+np.sum(self.V**2)/2))
        self.lambda_a = np.random.gamma(shape = 1 + self.N_users/2, scale = 1.0/(1+np.sum(self.a**2)/2))
        self.lambda_b = np.random.gamma(shape = 1 + self.N_films/2, scale = 1.0/(1+np.sum(self.b**2)/2))
        return
            
        
    def logprior(self, theta):
        """ logarithm of the prior distribution, which is a Gaussian distribution
            of variance varTheta
        Args:
            theta: parameter of R^d where the log prior is evaluated
        Returns:
            real value of the log prior evaluated at theta
        """
        return -(1. / (2*self.varTheta))* np.linalg.norm(theta)**2  \
                - (self.d/2.)*np.log(2*np.pi*self.varTheta)
    
    def gradlogprior(self, theta):
        """ gradient of the logarithm of the prior distribution, which is 
            a Gaussian distribution of variance varTheta
        Args:
            theta: parameter of R^d where the gradient log prior is evaluated
        Returns:
            R^d vector of the gradient of the log prior evaluated at theta
        """
        return -(1. / self.varTheta)*theta
    
    def potential(self, theta):
        """ logarithm of the posterior distribution
        Args:
            theta: parameter of R^d where the log posterior is evaluated
        Returns:
            real value of the log posterior evaluated at theta
        """
        return self.loglikelihood(theta)+self.logprior(theta)
    
    def minus_potential(self,theta):
        """Actually, a very silly function. Will re-write it later
        """
        return -self.potential(theta)
    
    def gradpotential(self,theta):
        """full gradient of posterior
        """
        return self.gradloglikelihood_determ(theta) + self.gradlogprior(theta)
    
    def stoch_grad(self,theta):
        """compute gradient estimate as in SGLD
        """
        batch_inds = np.random.choice(self.p,self.batch_size)
        return self.ratio*self.gradloglikelihood_stochastic(theta,batch_inds) + self.gradlogprior(theta)
    
    def stoch_grad_fixed_point(self,theta):
        """compute gradient estimate as in SGLD with fixed-point regularization
        """ 
        batch_inds = np.random.choice(self.p,self.batch_size)
        prior_part = self.gradlogprior(theta)-self.gradlogprior(self.theta_star)
        like_part = self.ratio*(self.gradloglikelihood_stochastic(theta,batch_inds) - self.gradloglikelihood_stochastic(self.theta_star,batch_inds))
        return prior_part + like_part
    
    def grad_SAGA(self,theta):
        """compute gradient estimate in SGLD with SAGA variance reduction procedure
        Yet implemented in another class, needs merging
        """
        return None
    
    def gradpotential_deterministic(self,theta):
        """
        A bit strange implementation of always deterministic gradient, this one is needed for fixed point search
        """
        return -self.gradloglikelihood_determ(theta) - self.gradlogprior(theta)