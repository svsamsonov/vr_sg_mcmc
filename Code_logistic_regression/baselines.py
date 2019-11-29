import numpy as np
from numpy.fft import fft,ifft
import scipy.sparse as sparse
import scipy.stats as spstats
import copy

def standartize(X_train,X_test,intercept = True):
    """Whitens noise structure, covariates updated
    """
    X_train = copy.deepcopy(X_train)
    X_test = copy.deepcopy(X_test)
    if intercept:#adds intercept term
        X_train = np.concatenate((np.ones(X_train.shape[0]).reshape(X_train.shape[0],1),X_train),axis=1)
        X_test = np.concatenate((np.ones(X_test.shape[0]).reshape(X_test.shape[0],1),X_test),axis=1)
    d = X_train.shape[1]
    # Centering the covariates 
    means = np.mean(X_train,axis=0)
    if intercept:#do not subtract the mean from the bias term
        means[0] = 0.0
    # Normalizing the covariates
    X_train -= means
    Cov_matr = np.dot(X_train.T,X_train)
    U,S,V_T = np.linalg.svd(Cov_matr,compute_uv = True)
    Sigma_half = U @ np.diag(np.sqrt(S)) @ V_T
    Sigma_minus_half = U @ np.diag(1./np.sqrt(S)) @ V_T
    X_train = X_train @ Sigma_minus_half
    # The same for test sample
    X_test = (X_test - means) @ Sigma_minus_half
    return X_train,X_test   

def set_bn(n):
    """
    function that sets size of the window in BM,OBM,SV estimates;
    please, make changes only here to change them simulteneously
    """
    #return np.round(2*np.power(n,0.33)).astype(int)
    return 10

def set_function(f_type,traj,inds_arr,params):
    """Main function to be evaluated in case of logistic regression
    Args:
        f_type - one of "posterior_mean","posterior_ll_point","posterior_ll_mean"
        traj - list of trajectories
        inds_arr - reasonable in case of "posterior_mean", otherwise ignored
        params - dictionary with fields "X","Y"
    returns:
        array of function values of respective shapes
    """
    if f_type == "posterior_mean":#params is ignored in this case
        f_vals = np.zeros((len(traj),len(traj[0]),len(inds_arr)),dtype = float)
        for traj_ind in range(len(traj)):
            for point_ind in range(len(inds_arr)):
                f_vals[traj_ind,:,point_ind] = set_f(traj[traj_ind],inds_arr[point_ind])
                
    elif f_type == "posterior_prob_point":
        f_vals = np.zeros((len(traj),len(traj[0]),len(inds_arr)),dtype = float)
        for traj_ind in range(len(traj)):
            for point_ind in range(len(inds_arr)):
                f_vals[traj_ind,:,point_ind] = set_f_point_prob(traj[traj_ind],params,inds_arr[point_ind])  
                
    elif f_type == "posterior_ll_point":#evaluate log-probabilies at one point
        f_vals = np.zeros((len(traj),len(traj[0]),len(params["X"])),dtype = float)
        for traj_ind in range(len(traj)):
            for point_ind in range(len(params["X"])):
                f_vals[traj_ind,:,point_ind] = set_f_point_ll(traj[traj_ind],params,inds_arr[point_ind])
                
    elif f_type == "posterior_prob_mean":
        f_vals = np.zeros((len(traj),len(traj[0]),1),dtype = float)
        for traj_ind in range(len(traj)):
            f_vals[traj_ind,:,0] = set_f_average_prob(traj[traj_ind],params)
            
    elif f_type == "posterior_prob_mean_probit":
        f_vals = np.zeros((len(traj),len(traj[0]),1),dtype = float)
        for traj_ind in range(len(traj)):
            f_vals[traj_ind,:,0] = set_f_average_prob_probit(traj[traj_ind],params)
            
    elif f_type == "posterior_prob_variance":
        f_vals = np.zeros((len(traj),len(traj[0]),1),dtype = float)
        for traj_ind in range(len(traj)):
            f_vals[traj_ind,:,0] = set_f_average_var(traj[traj_ind],params)
            
    elif f_type == "posterior_ll_mean":#evaluate average log-probabilities over test set
        f_vals = np.zeros((len(traj),len(traj[0]),1),dtype = float)
        for traj_ind in range(len(traj)):
            f_vals[traj_ind,:,0] = set_f_average_ll(traj[traj_ind],params)
            
    elif f_type == "success_prob_point":#success probabilities at given points
        f_vals = np.zeros((len(traj),len(traj[0]),len(inds_arr)),dtype = float)
        for traj_ind in range(len(traj)):
            for point_ind in range(len(inds_arr)):
                f_vals[traj_ind,:,point_ind] = set_f_success_point(traj[traj_ind],params,inds_arr[point_ind])
                
    elif f_type == "success_prob_mean":#success probabilities averaged
        f_vals = np.zeros((len(traj),len(traj[0]),1),dtype = float)
        for traj_ind in range(len(traj)):
            f_vals[traj_ind,:,0] = set_f_success_mean(traj[traj_ind],params)    
            
    elif f_type == "success_prob_varaince":#variance estimate for success probabilities
        f_vals = np.zeros((len(traj),len(traj[0]),1),dtype = float)
        for traj_ind in range(len(traj)):
            f_vals[traj_ind,:,0] = set_f_success_variance(traj[traj_ind],params)    
            
    else:#smthing strange
        raise "Not implemented error in set_function: check f_type value"
    return f_vals

def set_f(X,ind):
    """
    Element-wise function of observation, depending on ind, please, change it only here
    Arguments:
        X - np.array of shape (n,d);
        ind - int, 0 <= ind <= d 
    """
    return copy.deepcopy(X[:,ind])

def set_f_point_prob(X,params,ind):
    obs = params["X"][ind,:]
    Y = params["Y"][ind]
    #return 1./(1+np.exp(-X @ obs))
    return 1./(1+np.exp((1-2*Y)*X @ obs))

def set_f_point_ll(X,params,ind):
    """Function to compute point-wise test log-probabilities log p(y|x)
    Args:
        params - dict, defined in main notebook
    """
    obs = params["X"][ind,:]
    Y = params["Y"][ind]
    return -Y*np.log(1+np.exp(-X @ obs)) - (1-Y)*np.log(1+np.exp(X @ obs))

def set_f_average_prob(X,params):
    obs = params["X"]
    Y = params["Y"]
    return np.mean(np.divide(1.,1.+np.exp(np.dot(X,(obs*(1-2*Y).reshape(len(Y),1)).T))),axis=1)

def set_f_average_prob_probit(X,params):
    obs = params["X"]
    Y = params["Y"]
    return np.mean(spstats.norm.cdf(np.dot(X,(obs*(2*Y-1).reshape(len(Y),1)).T)),axis=1)

def set_f_average_var(X,params):
    obs = params["X"]
    Y = params["Y"]
    likelihoods = np.divide(1.,1.+np.exp(np.dot(X,(obs*(1-2*Y).reshape(len(Y),1)).T)))
    return np.mean(likelihoods**2,axis=1) - (np.mean(likelihoods,axis=1))**2

def set_f_average_ll(X,params):
    """Function to compute average test log-probabilities log p(y|x)
    """
    obs = params["X"]
    Y = params["Y"]
    return np.mean(-Y.reshape((1,len(Y)))*np.log(1+np.exp(-np.dot(X,obs.T))) -\
                              (1-Y).reshape(1,(len(Y)))*np.log(1+np.exp(np.dot(X,obs.T))),axis=1)

def set_f_success_point(X,params,ind):
    """Function to evaluate probability of success for a given vector X
    """
    obs = params["X"][ind,:]
    return 1./(1.+np.exp(-X@obs))
    
def set_f_success_mean(X,params):
    """Function to evaluate probability of success for a given vector X
    """
    obs = params["X"]
    Y = params["Y"]
    return np.mean(np.divide(1.,1.+np.exp(-np.dot(X,obs.T))),axis=1)

def qform_q(A,X,Y):
    """
    quickest way which I find to compute for each index i quadratic form <Ax,y> for each x = X[ind,:], y = Y[ind,:]
    arguments:
        A - np.array of shape (d,d)
        X,Y - np.array of shape (n,d)
    returns:
        np.array of shape (n)
    """
    return (X.dot(A)*Y).sum(axis=1)

def PWP(x,W):
    """
    performs multiplication (slow) with P - projector, W - topelitz (bn-diagonal) matrix
    Args:
        W - bn-diagonal matrix os shap (n,n) in csr format;
    returns:
        np.array of shape (n,) - result of PWP multiplicaton;
    """
    y = W @ (x - np.mean(x))
    return y - np.mean(y)

def mult_W(x,c):
    """
    performs multiplication (fast, by FFT) with W - toeplitz (bn-diagonal) matrix
    Args:
        c - vector of 
    returns:
        matvec product;
    """
    n = len(x)
    x_emb = np.zeros(2*n-1)
    x_emb[:n] = x
    return ifft(fft(c)*fft(x_emb)).real[:n]
    
def PWP_fast(x,c):
    """
    Same PWP as above, but now with FFT
    """
    y = mult_W(x - np.mean(x),c)
    return y - np.mean(y)

def Spectral_var(Y,W):
    """
    Compute spectral variance estimate for asymptotic variance with given kernel W for given vector Y
    """
    n = len(Y)
    return np.dot(PWP_fast(Y,W),Y)/n
                  
def simple_funcs(X,ind):
    """
    """
    if ind == 0:
        return np.cos(X.sum(axis=1))
    elif ind == 1:
        return np.cos((X**2).sum(axis=1))

def init_samples(X):
    """
    initialize sample matrix for 
    """
    samples = np.zeros_like(X)
    for ind in range(X.shape[1]):
        samples[:,ind] = set_f(X,ind)
    return samples

def construct_ESVM_kernel(n):
    """
    Same as before, but now returns only first row of embedding circulant matrix;
    Arguments:
        n - int,size of the matrix;
    Returns:
        c - np.array of size (2n-1);
    """
    bn = set_bn(n)
    trap_left = np.linspace(0,1,bn)
    trap_center = np.ones(2*bn+1,dtype = float)
    trap_right = np.linspace(1,0,bn)
    diag_elems = np.concatenate([trap_left,trap_center,trap_right])
    c = np.zeros(2*n-1,dtype = np.float64)
    c[0:bn+1] = 1.0
    c[bn+1:2*bn+1] = trap_right
    c[-bn:] = 1.0
    c[-2*bn:-bn] = trap_left
    return c
        
def split_dataset(X,Y,test_size):
    """Implements (a bit strange) splitting of train dataset at test and train part
    Args:
        test_size - number of pairs (X,Y) to report in test; 
    Return:
        ...
    """
    np.random.seed(1814)
    batch_inds = np.random.choice(len(X),size = test_size,replace=False)
    X_test = copy.deepcopy(X[batch_inds,:])
    Y_test = Y[batch_inds]
    X_train = np.delete(X,batch_inds,0)
    mask = np.ones(len(Y),dtype = bool)
    mask[batch_inds] = False
    Y_train = Y[mask]
    return X_train,Y_train,X_test,Y_test  
    
    
