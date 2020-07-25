import numpy as np
from baselines import set_function, Spectral_var

def ZVpolyOne(traj, traj_grad, f_target, params, W_spec):
    n, d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    cov1 = np.cov(traj_grad, rowvar=False)
    if d==1 :
        A = 1/cov1
    else:
        A = np.linalg.inv(cov1)
    covariance = np.cov(np.concatenate((-traj_grad, samples), axis=1), rowvar=False)
    paramZV1 = -np.dot(A,covariance[:d, d:])
    #print("ZV1: ",paramZV1)
    ZV1 = samples - np.dot(traj_grad, paramZV1)
    mean_ZV1 = np.mean(ZV1, axis = 0)
    var_ZV1 = Spectral_var(ZV1[:,0],W_spec)
    return mean_ZV1, var_ZV1

def ZVpolyTwo(traj, traj_grad, f_target, params, W_spec):
    n, d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    Lpoisson = np.zeros((n,int(d*(d+3)/2)))
    Lpoisson[:,np.arange(d)] = - traj_grad
    Lpoisson[:,np.arange(d, 2*d)] = 2*(1. - np.multiply(traj, traj_grad))
    k=2*d
    for j in np.arange(d-1):
        for i in np.arange(j+1,d):
            Lpoisson[:,k] = -np.multiply(traj_grad[:,i], traj[:,j]) \
                    -np.multiply(traj_grad[:,j], traj[:,i])
            k=k+1
    cov1 = np.cov(Lpoisson, rowvar=False)
    if cov1.shape[0] == 1:
        A = 1/cov1
    else:
        A = np.linalg.inv(cov1)
    cov2 = np.cov(np.concatenate((Lpoisson, samples),axis=1), rowvar=False)
    B = cov2[0:int(d*(d+3)/2), int(d*(d+3)/2):]
    paramZV2 = - np.dot(A,B)
    ZV2 = samples + np.dot(Lpoisson, paramZV2)
    mean_ZV2 = np.mean(ZV2, axis = 0)
    var_ZV2 = Spectral_var(ZV2[:,0],W_spec)
    return mean_ZV2, var_ZV2

def CVpolyOne(traj,traj_grad, f_target, params, W_spec):
    n, d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    #covariance = np.cov(np.concatenate((traj, samples), axis=1), rowvar=False)
    #paramCV1 = covariance[:d, d:]
    paramCV1 = (np.transpose(traj)@(samples - np.mean(samples)))/traj.shape[0]
    print("CV1: ",paramCV1)
    CV1 = samples - np.dot(traj_grad, paramCV1)
    mean_CV1 = np.mean(CV1, axis = 0)
    var_CV1 = Spectral_var(CV1[:,0],W_spec)
    return mean_CV1, var_CV1

def CVpolyOneGaussian(traj,traj_grad, f_target, params, W_spec):
    """
    Version of CV's with family of $\psi$ given by 2-dimensional gaussians
    """
    n, d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    print(samples.shape)
    print(traj.shape)
    covariance = np.cov(np.concatenate((traj, samples), axis=1), rowvar=False)
    paramCV1 = covariance[:d, d:]
    CV1 = samples - np.dot(traj_grad, paramCV1)
    mean_CV1 = np.mean(CV1, axis = 0)
    var_CV1 = Spectral_var(CV1[:,0],W_spec)
    return mean_CV1, var_CV1

def CVpolyOneUpdated(traj,traj_grad, f_target, params, W_spec):
    n, d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    print(samples.shape)
    #subtract mean
    samples = samples - np.mean(samples)
    print(traj.shape)
    #covariance = np.cov(np.concatenate((traj, samples), axis=1), rowvar=False)
    #paramCV1 = covariance[:d, d:]
    paramCV1 = (np.transpose(traj)@ samples)/traj.shape[0]
    CV1 = samples - np.dot(traj_grad, paramCV1)
    mean_CV1 = np.mean(CV1, axis = 0)
    var_CV1 = Spectral_var(CV1[:,0],W_spec)
    return mean_CV1, var_CV1

def CVpolyTwo(traj, traj_grad, f_target, params, W_spec):
    n, d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    poisson = np.zeros((n,int(d*(d+3)/2)))
    poisson[:,np.arange(d)] = traj
    poisson[:,np.arange(d, 2*d)] = np.multiply(traj, traj)
    k = 2*d
    for j in np.arange(d-1):
        for i in np.arange(j+1,d):
            poisson[:,k] = np.multiply(traj[:,i], traj[:,j])
            k=k+1
    Lpoisson = np.zeros((n,int(d*(d+3)/2)))
    Lpoisson[:,np.arange(d)] = - traj_grad
    Lpoisson[:,np.arange(d, 2*d)] = 2*(1. - np.multiply(traj, traj_grad))
    k=2*d
    for j in np.arange(d-1):
        for i in np.arange(j+1,d):
            Lpoisson[:,k] = -np.multiply(traj_grad[:,i], traj[:,j]) \
                    -np.multiply(traj_grad[:,j], traj[:,i])
            k=k+1
    
    cov1 = np.cov(np.concatenate((poisson, -Lpoisson), axis=1), rowvar=False)
    A = np.linalg.inv(cov1[0:int(d*(d+3)/2), int(d*(d+3)/2):d*(d+3)])
    cov2 = np.cov(np.concatenate((poisson, samples),axis=1), rowvar=False)
    B = cov2[0:int(d*(d+3)/2), int(d*(d+3)/2):]
    paramCV2 = np.dot(A,B)
    CV2 = samples + np.dot(Lpoisson, paramCV2)
    mean_CV2 = np.mean(CV2, axis = 0)
    var_CV2 = Spectral_var(CV2[:,0],W_spec)
    return mean_CV2,var_CV2

def GausCV(traj,sample):
    """
    returns matrix of gaussian CV's 
    """
    #m=7 - good
    m=10
    pen=0.
    x = np.linspace(-5,5,m)
    y = np.linspace(-5,5,m)
    sigma_squared = 3.0
    xx, yy = np.meshgrid(x,y)
    d = m**2
    #print(xx)
    mu = np.concatenate((xx.reshape((-1,1)),yy.reshape((-1,1))),axis=1)
    #traj = np.repeat(traj[:,np.newaxis,:], 25, axis=1)
    #print(traj.shape)
    traj_adj = (np.repeat(traj[:,np.newaxis,:], d, axis=1)-mu[np.newaxis,:])/sigma_squared
    #print(traj_adj.shape)
    psi_matr = np.zeros((traj.shape[0],d))
    for i in range(d):
        psi_matr[:,i] = np.exp(-np.sum((traj-mu[i].reshape((1,2)))**2, axis=1)/(2*sigma_squared))
    #print(psi_matr.shape)
    cov = np.dot(sample.T - sample.mean(), psi_matr - psi_matr.mean(axis=0)) / traj.shape[0]
    print(cov.shape)
    jac_matr = -traj_adj*(psi_matr.reshape((psi_matr.shape[0],psi_matr.shape[1],1)))
    H = np.mean(np.matmul(jac_matr,jac_matr.transpose(0,2,1)),axis=0)
    param_CV = np.linalg.inv(H + pen*np.eye(H.shape[0])).dot(cov.T)
    print(np.sqrt(np.sum(param_CV**2)))
    jac_star = np.sum(jac_matr*param_CV[np.newaxis,:],axis=1)
    delta_star = (psi_matr*(np.sum(traj_adj**2, axis=2)-traj.shape[1]/sigma_squared)).dot(param_CV)
    return jac_star,delta_star

def TryCV(traj,sample):
    """
    returns matrix of gaussian CV's 
    """
    d = 100
    x = np.linspace(-5,5,10)
    y = np.linspace(-5,5,10)
    sigma_squared = 1.0
    xx, yy = np.meshgrid(x,y)
    #print(xx)
    #xx = np.ravel(xx)
    #yy = np.ravel(yy)
    mu = np.concatenate((xx.reshape((-1,1)),yy.reshape((-1,1))),axis=1)
    print(mu.shape)
    #print(mu.shape)
    #traj = np.repeat(traj[:,np.newaxis,:], 25, axis=1)
    #print(traj.shape)
    #d = 50
    #mu = np.random.randn(d,2)
    #print(mu.shape)
    jac_star = np.zeros((traj.shape[0],traj.shape[1]))
    delta = np.zeros(traj.shape[0])
    for i in range(d):
        a = np.exp(-np.sum((traj-mu[i].reshape((1,2)))**2, axis=1)/(2*sigma_squared))
        print(a.shape)
        jac_star -= (traj-mu[i].reshape((1,2)))*np.exp(-np.sum((traj-mu[i].reshape((1,2)))**2, axis=1)/(2*sigma_squared)).reshape((-1,1))
        delta += (np.sum((traj-mu[i].reshape((1,2)))**2, axis=1)/(sigma_squared**2)-traj.shape[1])*np.exp(-np.sum((traj-mu[i].reshape((1,2)))**2, axis=1)/(2*sigma_squared))
    #jac_star = np.sum(jac_matr,axis=1)
    #delta_star = (psi_matr*(np.sum(traj_adj**2, axis=2)-traj.shape[1]/sigma_squared)).dot(param_CV)
    return jac_star,delta

def CVpolyGaussian(traj, traj_grad, f_target, params, W_spec):
    n,d = traj.shape
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    #jac,delta = TryCV(traj,samples)
    jac,delta = GausCV(traj,samples)
    #print(jac.shape)
    #print(delta.shape)
    CV = samples - np.sum(traj_grad*jac, axis = 1).reshape((-1,1)) + delta
    #CV = -np.sum(traj_grad*jac, axis = 1).reshape((-1,1)) + delta.reshape((-1,1))
    mean_CV = np.mean(CV, axis = 0)
    #print(mean_CV)
    var_CV = Spectral_var(CV[:,0],W_spec)
    return mean_CV, var_CV
    
def Eval_ZVCV_Gaus(traj,traj_grad, f_target, params, W_spec):
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    mean_vanilla = np.mean(samples)
    vars_vanilla = Spectral_var(samples[:,0],W_spec)
    mean_ZV, var_ZV = ZVpolyOne(traj,traj_grad,f_target,params,W_spec)
    #mean_CV, var_CV = CVpolyOne(traj,traj_grad,f_target,params,W_spec)
    mean_CV, var_CV = CVpolyGaussian(traj,traj_grad,f_target,params,W_spec)
    return (mean_vanilla,mean_ZV,mean_CV), (vars_vanilla,var_ZV,var_CV)

def Eval_ZVCV(traj,traj_grad, f_target, params, W_spec):
    if f_target == "sum":
        samples = traj.sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_comps":
        samples = traj[:,params["ind"]].reshape(-1,1)
    elif f_target == "sum_comps_squared":
        samples = np.square(traj[:,params["ind"]]).reshape(-1,1)
    elif f_target == "sum_squared":
        samples = np.square(traj).sum(axis = 1).reshape(-1,1)
    elif f_target == "sum_4th":
        samples = ((traj)**4).sum(axis = 1).reshape(-1,1)
    elif f_target == "exp_sum":
        samples = np.exp(traj.sum(axis = 1)).reshape(-1,1)
    else:
        traj = np.expand_dims(traj, axis = 0)
        samples = set_function(f_target,traj,[0],params)
        traj = traj[0]
        samples = samples[0]
    mean_vanilla = np.mean(samples)
    vars_vanilla = Spectral_var(samples[:,0],W_spec)
    mean_ZV1, var_ZV1 = ZVpolyOne(traj,traj_grad,f_target,params,W_spec)
    mean_ZV2, var_ZV2 = ZVpolyTwo(traj,traj_grad,f_target,params,W_spec)
    #mean_CV1, var_CV1 = CVpolyOneUpdated(traj,traj_grad,f_target,params,W_spec)
    mean_CV1, var_CV1 = CVpolyOne(traj,traj_grad,f_target,params,W_spec)
    mean_CV2, var_CV2 = CVpolyTwo(traj,traj_grad,f_target,params,W_spec)
    return (mean_vanilla,mean_ZV1, mean_ZV2, mean_CV1, mean_CV2), (vars_vanilla, var_ZV1, var_ZV2, var_CV1, var_CV2)    
    