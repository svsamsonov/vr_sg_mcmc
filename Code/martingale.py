import numpy as np
import numpy.polynomial as P
import scipy as sp
from sklearn.preprocessing import PolynomialFeatures
from samplers import ULA_light
from potentials import GaussPotential,GaussMixture,GausMixtureIdent,GausMixtureSame
import copy
from baselines import set_function
import time

def H(k, x):
    if k==0:
        return 1.0
    if k ==1:
        return x
    if k==2:
        return (x**2 - 1)/np.sqrt(2)
    c = np.zeros(k+1,dtype = float)
    c[k] = 1.0
    h = P.hermite_e.hermeval(x,c) / np.sqrt(sp.special.factorial(k)) 
    return h

def split_index(k,d,max_deg):
    """
    transforms single index k into d-dimensional multi-index d with max_deg degree each coordinate at most;
    """
    k_vec = np.zeros(d,dtype = int)
    for i in range(d):
        k_vec[-(i+1)] = k % (max_deg + 1)
        k = k // (max_deg + 1)
    return k_vec

def hermite_val(k_vec,x_vec):
    P = 1.0
    d = x_vec.shape[0]
    for i in range(d):
        P = P * H(k_vec[i],x_vec[i])
    return P

def eval_hermite(k,x_vec,max_deg):
    """
    Evaluates Hermite polynomials at component x_vec by multi-index obtained from single integer k;
    Args:
        max_deg - integer, maximal degree of a polynomial at fixed dimension component;
        k - integer, number of given basis function; 1 <= k <= (max_deg+1)**d
        x_vec - np.array of shape(d,N), where d - dimension, N - Train (or Test) sample size
    """
    k_vec = split_index(k,len(x_vec),max_deg)
    #now we initialised k_vec
    return hermite_val(k_vec,x_vec)

def approx_q(X_train,Y_train,N_traj_train,lag,max_deg):
    """
    Function to regress q functions on a polynomial basis;
    Args:
        X_train - train tralectory;
        Y_train - function values;
        N_traj_train - number of training trajectories;
        lag - truncation point for coefficients, those for |p-l| > lag are set to 0;
        max_deg - maximum degree of polynomial in regression
    """
    dim = X_train[0,:].shape[0]
    print("dimension = ",dim)
    coefs_poly = np.array([])
    for i in range(lag):
        x_all = np.array([])
        y_all = np.array([])
        for j in range(N_traj_train):
            y = Y_train[j,i:,0]
            if i == 0:
                x = X_train[j,:]
            else:
                x = X_train[j,:-i]
            #concatenate results
            if x_all.size == 0:
                x_all = x
            else:
                x_all = np.concatenate((x_all,x),axis = 0)
            y_all = np.concatenate([y_all,y])
        #should use polyfeatures here
        print("variance: ",np.var(y_all))
        print(y_all[:50])
        poly = PolynomialFeatures(max_deg)
        X_features = poly.fit_transform(x_all)
        print(X_features.shape)
        lstsq_results = np.linalg.lstsq(X_features,y_all,rcond = None)
        coefs = copy.deepcopy(lstsq_results[0])
        coefs.resize((1,X_features.shape[1]))           
        if coefs_poly.size == 0:
            coefs_poly = copy.deepcopy(coefs)
        else:
            coefs_poly = np.concatenate((coefs_poly,coefs),axis=0)
    return coefs_poly

def approx_q_independent(X_train,Y_train,N_traj_train,lag,max_deg):
    """
    Function to regress q functions bases on a polynomial basis and big number of short independent trajectories
    """
    dim = X_train[0,:].shape[0]
    print("dimension = ",dim)
    coefs_poly = np.array([])
    for i in range(lag):
        x_all = X_train[:,0,:]
        y_all = Y_train[:,i,0]
        #should use polyfeatures here
        poly = PolynomialFeatures(max_deg)
        X_features = poly.fit_transform(x_all)
        print(X_features.shape)
        lstsq_results = np.linalg.lstsq(X_features,y_all,rcond = None)
        coefs = copy.deepcopy(lstsq_results[0])
        coefs.resize((1,X_features.shape[1]))           
        if coefs_poly.size == 0:
            coefs_poly = copy.deepcopy(coefs)
        else:
            coefs_poly = np.concatenate((coefs_poly,coefs),axis=0)
    return coefs_poly

def get_indices_poly(ind,K_max,S_max):
    """
    Transforms 1d index into 2d index
    """
    S = ind % (S_max + 1)
    K = ind // (S_max + 1) 
    return K,S

def init_basis_polynomials(K_max,S_max,st_norm_moments,gamma):
    """
    Represents E[H_k(xi)*(x-gamma mu(x) + sqrt{2gamma}xi)^s] as a polynomial of variable $y$, where y = x - gamma*mu(x)
    Args:
        K_max - maximal degree of Hermite polynomial;
        S_max - maximal degree of regressor polynomial;
        st_norm_moments - array containing moments of standard normal distribution;
    Return:
        Polynomial coefficients
    """
    Poly_coefs_regression = np.zeros((K_max+1,S_max+1,S_max+1),dtype = float)
    for k in range(Poly_coefs_regression.shape[0]):
        for s in range(Poly_coefs_regression.shape[1]):
            herm_poly = np.zeros(K_max+1, dtype = float)
            herm_poly[k] = 1.0
            herm_k = P.hermite_e.herme2poly(herm_poly)
            herm_k = herm_k / np.sqrt(sp.special.factorial(k))
            c = np.zeros(S_max+1, dtype = float)
            for deg in range(s+1):
                c[deg] = (np.sqrt(2*gamma)**(s-deg))*sp.special.binom(s,deg)*np.dot(herm_k,st_norm_moments[(s-deg):(s - deg+len(herm_k))])
            Poly_coefs_regression[k,s,:] = c
    return Poly_coefs_regression

def init_moments(order):
    """
    Compute moments of standard normal distribution
    """
    moments_stand_norm = np.zeros(2*order+1,dtype = float)
    moments_stand_norm[0] = 1.0
    moments_stand_norm[1] = 0.0
    for i in range(len(moments_stand_norm)-2):
        moments_stand_norm[i+2] = sp.special.factorial2(i+1, exact=False)
    #eliminate odd
    moments_stand_norm[1::2] = 0
    return moments_stand_norm

def generate_combines(k,l):
    """
    function that returns array of combinations of given cardinality
    """
    return 0
    
def get_representations(k,s,d,K_max):
    """
    Factorizes k and s into a d-vector of different dimensions
    Args:
        k - hermite polynomial number;
        s - number of basis functions;
        d - dimension
    """
    k_vec = np.zeros(d,dtype = int)
    s_vec = np.zeros(d,dtype = int)
    #initialize k_vec
    for ind in range(d):
        k_vec[-(ind+1)] = k % (K_max + 1)
        k = k // (K_max + 1)
    #initialize s_vec
    if d == 1:
        vec_table = np.array([[0],[1],[2],[3],[4],[5]])
        s_vec = vec_table[s,:]
    elif d == 2:
        vec_table = np.array([[0,0],[1,0],[0,1],[2,0],[1,1],[0,2],[3,0],[2,1],[1,2],[0,3],[4,0],[3,1],[2,2],[1,3],[0,4]])
        s_vec = vec_table[s,:]
    elif d == 4:
        vec_table = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                              [2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],
                              [0,2,0,0],[0,1,1,0],[0,1,0,1],
                              [0,0,2,0],[0,0,1,1],
                              [0,0,0,2]])
        s_vec = vec_table[s,:]
    else:
        vec_table = np.zeros((d+1,d),dtype = int)
        for ind in range(1,len(vec_table)):
            vec_table[ind,ind-1] = 1
        s_vec = vec_table[s,:]
    """
    elif d == 4:
        vec_table = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                              [2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],
                              [0,2,0,0],[0,1,1,0],[0,1,0,1],
                              [0,0,2,0],[0,0,1,1],
                              [0,0,0,2]]) 
    """
    return k_vec,s_vec

def test_traj(Potential,coefs_poly_regr,step,r_seed,lag,K_max,S_max,N_burn,N_test,d,f_type,inds_arr,params,x0,fixed_start):
    """
    """
    X_test,Noise = ULA_light(r_seed,Potential,step, N_burn, N_test, d, return_noise = True, x0 = x0, fixed_start = fixed_start)
    print(X_test[0])
    Noise = Noise.T
    test_stat_vanilla = np.zeros(N_test,dtype = float)
    test_stat_vr = np.zeros_like(test_stat_vanilla)
    #compute number of basis polynomials
    num_basis_funcs = (K_max+1)**d
    #print("number of basis functions = ",num_basis_funcs)
    #compute polynomials of noise variables Z_l
    poly_vals = np.zeros((num_basis_funcs,N_test),dtype = float)
    for k in range(len(poly_vals)):
        poly_vals[k,:] = eval_hermite(k,Noise,K_max)
    #print(poly_vals.shape)
    #initialize function
    #f_vals_vanilla = np.sum(X_test,axis=1)
    #f_vals_vanilla = X_test[:,0]
    f_vals_vanilla = set_function(f_type,np.expand_dims(X_test, axis=0),inds_arr,params)
    f_vals_vanilla = f_vals_vanilla[0,:,0]
    cvfs = np.zeros_like(f_vals_vanilla)
    st_norm_moments = init_moments(K_max+S_max+1)
    table_coefs = init_basis_polynomials(K_max,S_max,st_norm_moments,step)
    #print(table_coefs.shape)
    start_time = time.time()
    for i in range(1,len(cvfs)):
        #start computing a_{p-l} coefficients
        num_lags = min(lag,i)
        a_vals = np.zeros((num_lags,num_basis_funcs),dtype = float)#control variates
        for func_order in range(num_lags):#for a fixed lag Q function
            #compute \hat{a} with fixed lag
            x = X_test[i-1-func_order]
            x_next = x + step*Potential.gradpotential(x)
            for k in range(1,num_basis_funcs):
                a_cur = np.ones(coefs_poly_regr.shape[1], dtype = float)
                for s in range(len(a_cur)):
                    k_vect,s_vect = get_representations(k,s,d,K_max)
                    #print("K = ",k_vect)
                    #print("S = ",s_vect)
                    for dim_ind in range(d):
                        a_cur[s] = a_cur[s]*P.polynomial.polyval(x_next[dim_ind],table_coefs[k_vect[dim_ind],s_vect[dim_ind],:])
                a_vals[-(func_order+1),k] = np.dot(a_cur,coefs_poly_regr[func_order,:])
            #OK, now I have coefficients of the polynomial, and I need to integrate it w.r.t. Gaussian measure
        #print("sum of coefficients",np.sum(np.abs(a_vals)))
        #print(a_vals)
        cvfs[i] = np.sum(a_vals*(poly_vals[:,i-num_lags+1:i+1].T))
        #save results
        test_stat_vanilla[i] = np.mean(f_vals_vanilla[1:(i+1)])
        test_stat_vr[i] = test_stat_vanilla[i] - np.sum(cvfs[1:(i+1)])/i
    end_time = time.time() - start_time
    return test_stat_vanilla, test_stat_vr

def test_monte_carlo(r_seed,Potential,step,N_burn,N,d,return_noise, x0, fixed_start):
    """
    Function to test vanilla mcmc with large sample sizes
    """
    X_test = ULA_light(r_seed,Potential,step, N_burn, N, d, return_noise, x0, fixed_start)
    return X_test