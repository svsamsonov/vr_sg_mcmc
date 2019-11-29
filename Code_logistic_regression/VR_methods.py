import numpy as np
from baselines import PWP_fast,Spectral_var,qform_q

#first-order control variates: ZAV, ZV, LS
def qform_1_ESVM(a,f_vals,X_grad,W,ind,n):
    """
    ESVM quadratic form computation: asymptotic variance estimator based on kernel W; 
    """
    x_cur = f_vals[:,ind] + X_grad @ a
    return Spectral_var(x_cur,W)

def grad_qform_1_ESVM(a,f_vals,X_grad,W,ind,n):
    """
    gradient of ESVM quadratic form
    """
    Y = f_vals[:,ind] + X_grad @ a
    return 2./n * (X_grad*PWP_fast(Y,W).reshape((n,1))).sum(axis=0)

def qform_1_EVM(a,f_vals,X_grad,ind,n):
    """
    Least squares evaluated for EVM-1
    """
    x_cur = f_vals[:,ind] + X_grad @ a
    return 1./(n-1)*np.dot(x_cur - np.mean(x_cur),x_cur - np.mean(x_cur))

def grad_qform_1_EVM(a,f_vals,X_grad,ind,n):
    """
    Gradient for quadratic form in EVM-1 method 
    """
    Y = f_vals[:,ind] + X_grad @ a
    return 2./(n-1) * (X_grad*(Y - np.mean(Y)).reshape((n,1))).sum(axis=0)

def qform_1_LS(a,f_vals,X_grad,ind,n):
    """
    Least Squares-based control variates
    """
    x_cur = f_vals[:,ind] + X_grad @ a
    return np.mean(x_cur**2)
    
def grad_qform_1_LS(a,f_vals,X_grad,ind,n):
    """
    Gradient for Least-Squares control variates
    """
    Y = f_vals[:,ind] + X_grad @ a
    return 2./n * X_grad.T @ Y

#################################################################################################################################
#second-order control variates: ESVM, EVM, LS
def qform_2_ESVM(a,f_vals,X,X_grad,W,ind,n,alpha = 0.0):
    """
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    """
    d = X_grad.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    x_cur = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    return Spectral_var(x_cur,W) + alpha*np.sum(B**2)

def grad_qform_2_ESVM(a,f_vals,X,X_grad,W,ind,n, alpha = 0.0):
    """
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    """
    d = X_grad.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    Y = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    #gradient w.r.t. b
    nabla_b = 2./n * (X_grad*PWP_fast(Y,W).reshape((n,1))).sum(axis=0)
    #gradient w.r.t B
    nabla_f_B = np.matmul(X_grad.reshape((n,d,1)),X.reshape((n,1,d)))
    nabla_f_B = nabla_f_B + nabla_f_B.transpose((0,2,1)) + 2*np.eye(d).reshape((1,d,d))                     
    nabla_B = 2./n*np.sum(nabla_f_B*PWP_fast(Y,W).reshape((n,1,1)),axis = 0)
    #add ridge
    nabla_B += 2*alpha*B
    #stack gradients together
    grad = np.zeros((d+1)*d,dtype = np.float64)
    grad[:d] = nabla_b
    grad[d:] = nabla_B.ravel()
    return grad
    

def qform_2_EVM(a,f_vals,X,X_grad,ind,n):
    """
    Least squares evaluated for EVM-2 method
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    Returns:
        function value for index ind, scalar variable
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    x_cur = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    return 1./(n-1)*np.dot(x_cur - np.mean(x_cur),x_cur - np.mean(x_cur))

def grad_qform_2_EVM(a,f_vals,X,X_grad,ind,n):
    """
    Gradient for quadratic form in EVM-2 method
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    Y = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    #gradient w.r.t. b
    nabla_b = 2./(n-1)*(X_grad*(Y - np.mean(Y)).reshape((n,1))).sum(axis=0)
    #gradient w.r.t B
    nabla_f_B = np.matmul(X_grad.reshape((n,d,1)),X.reshape((n,1,d)))
    nabla_f_B = nabla_f_B + nabla_f_B.transpose((0,2,1)) + 2*np.eye(d).reshape((1,d,d))                     
    nabla_B = 2./(n-1)*np.sum(nabla_f_B*(Y-np.mean(Y)).reshape((n,1,1)),axis = 0)
    #stack gradients together
    grad = np.zeros((d+1)*d,dtype = np.float64)
    grad[:d] = nabla_b
    grad[d:] = nabla_B.ravel()
    return grad

def qform_2_LS(a,f_vals,X,X_grad,ind,n):
    """
    Least squares evaluation for 2nd order polynomials as control variates;
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    Returns:
        function value for index ind, scalar variable
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    x_cur = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    return np.mean(x_cur**2)

def grad_qform_2_LS(a,f_vals,X,X_grad,ind,n):
    """
    Gradient for quadratic form in LS-2 method
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    Y = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    #gradient w.r.t. b
    nabla_b = 2./n * X_grad.T @ Y
    #gradient w.r.t B
    nabla_f_B = np.matmul(X_grad.reshape((n,d,1)),X.reshape((n,1,d)))
    nabla_f_B = nabla_f_B + nabla_f_B.transpose((0,2,1)) + 2*np.eye(d).reshape((1,d,d))                     
    nabla_B = 2./n*np.sum(nabla_f_B*Y.reshape((n,1,1)),axis = 0)
    #stack gradients together
    grad = np.zeros((d+1)*d,dtype = np.float64)
    grad[:d] = nabla_b
    grad[d:] = nabla_B.ravel()
    return grad

#################################################################################################################################
#wrappers for quadratic forms and their gradients calculations
#first-order methods
def train_1st_order(a,typ,f_vals,traj_grad_list,W,ind,n):
    """
    Universal wrapper for ESVM, EVM and LS quadratic forms
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(f_vals)
    val_list = np.zeros(n_traj)
    for i in range(len(val_list)):
        if typ == "ESVM":
            val_list[i] = qform_1_ESVM(a,f_vals[i],traj_grad_list[i],W,ind,n)
        elif typ == "EVM":
            val_list[i] = qform_1_EVM(a,f_vals[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            val_list[i] = qform_1_LS(a,f_vals[i],traj_grad_list[i],ind,n)
        else:
            raise "Not implemented error in Train_1st_order: something goes wrong"
    return np.mean(val_list)

def train_1st_order_grad(a,typ,f_vals,traj_grad_list,W,ind,n):
    """
    Universal wrapper for ESVM,EVM and LS quadratic forms gradients calculations
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(f_vals)
    grad_vals = np.zeros_like(a)
    for i in range(n_traj):
        if typ == "ESVM":
            grad_vals += grad_qform_1_ESVM(a,f_vals[i],traj_grad_list[i],W,ind,n)
        elif typ == "EVM":
            grad_vals += grad_qform_1_EVM(a,f_vals[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            grad_vals += grad_qform_1_LS(a,f_vals[i],traj_grad_list[i],ind,n)
    grad_vals /= n_traj
    return grad_vals

#second-order methods
def train_2nd_order(a,typ,f_vals,traj_list,traj_grad_list,W,ind,n,alpha):
    """
    average spectral variance estimation for given W matrix, based on len(traj_list) trajectories
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(traj_list)
    val_list = np.zeros(n_traj)
    for i in range(len(val_list)):
        if typ == "ESVM":
            val_list[i] = qform_2_ESVM(a,f_vals[i],traj_list[i],traj_grad_list[i],W,ind,n,alpha)
        elif typ == "EVM":
            val_list[i] = qform_2_EVM(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            val_list[i] = qform_2_LS(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
        else:
            raise "Not implemented error in Train_1st_order: something goes wrong"
    return np.mean(val_list)

def train_2nd_order_grad(a,typ,f_vals,traj_list,traj_grad_list,W,ind,n,alpha):
    """
    gradient for average SV estimate for given W matrix
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(traj_list)
    grad_vals = np.zeros_like(a)
    for i in range(n_traj):
        if typ == "ESVM":
            grad_vals += grad_qform_2_ESVM(a,f_vals[i],traj_list[i],traj_grad_list[i],W,ind,n,alpha)
        elif typ == "EVM":
            grad_vals += grad_qform_2_EVM(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            grad_vals += grad_qform_2_LS(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
    grad_vals /= n_traj
    return grad_vals