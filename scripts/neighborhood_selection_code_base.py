import numpy as np
from code_base import *


def EBIC_GaussianGM(S,Theta,n,gamma=0.5):
    
    p = np.shape(S)[0]
    
    L_n = n/2 * (- np.trace(np.matmul(S,Theta)) + np.log(np.linalg.det(Theta)) )
    
    off_diag_Theta =  Theta - np.diag(np.diag(Theta))
    
    Edge_Set_Length = np.sum(off_diag_Theta!=0) / 2
    
    EBIC = -2 * L_n + Edge_Set_Length * np.log(n) + 4 * Edge_Set_Length * gamma * np.log(p)
    
    if np.isnan(EBIC):
        EBIC = np.inf
        print("The determinant is",np.linalg.det(Theta))
        
    return(EBIC)



from numpy import linalg as LA

def refit_GaussianGM(S,W,rho=1):
    
    W[W != 0] = 1
    
    p = np.shape(S)[0]    
    MAX_ITER = 2000
    
    Z = np.ones([p,p])
    U = np.ones([p,p])
    obj = []
    
    for k in range(MAX_ITER):
        
        [L,Q] = LA.eig(rho*(Z - U) - S)
        
        Lambda = L
        xi = (Lambda + np.sqrt( np.power(Lambda,2) + 4*rho)) /(2*rho)
        X = Q @ np.diag(xi) @ Q.T
        
        Z = X + U
        Z[ W == 0 ] = 0
        
        U = U + (X - Z)
        
        obj.append(  np.trace(np.matmul(S,X)) - np.log(np.linalg.det(X)) + 1000 * np.sum( Z[W==0] !=0 ) )

        if ( (k > 1) and (abs(obj[k] - obj[k-1]) < 1e-3) ):
            break;
        elif k == (MAX_ITER-1):
            print("Algorithm does not converge")
    
    return(Z)








