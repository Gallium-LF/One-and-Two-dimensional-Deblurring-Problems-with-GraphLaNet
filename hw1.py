import numpy as np
import matplotlib.pyplot as plt

def k_function(t,s):
    if t<=s:
        return t*(s-1)
    else:
        return s*(t-1)
    
def x_function(s):
    return s

def midpoint_quadrature_rule(n,a,b,c,d):
    h1=(b-a)/n
    h2=(d-c)/n
    K=np.zeros((n,n),dtype=np.float32)
    x=np.zeros((n,1),dtype=np.float32)
    for j in range(0,n):
        s_j=a+(j+0.5)*h1
        x[j,0]=x_function(s_j)
        for i in range(0,n):
            t_i=c+(i+0.5)*h2
            K[i,j]=k_function(t_i,s_j)
    return h1*K,x

def plot_con_number():
    con_numbers =np.zeros((10,1),dtype=np.float32) #using an array to store the condition numbers
    for n in range(1,10):
        (K,x)=midpoint_quadrature_rule(n*10,0,1,0,1)
        eig=np.linalg.eigvals(K)
        con_numbers[n,0]=np.max(eig)/np.min(eig)
    fig , ax = plt.subplots()
    ax.scatter(np.arange(10)*10,con_numbers,alpha=0)
    plt.plot(np.arange(10)*10,con_numbers)
    plt.xlabel("n (number of subintervals)")
    plt.ylabel("conditioning number")
    plt.show()

def plot_x_dagger(n):
    (K,x)=midpoint_quadrature_rule(n,0,1,0,1)
    (u,s,v_T)=np.linalg.svd(K)
    s=s*np.eye(n,dtype=np.float32)
    y=K@x
    x_dagger=v_T.T@np.linalg.pinv(s)@u.T@y
    fig , ax = plt.subplots()
    ax.scatter(np.arange(n),x,alpha=0)
    ax.scatter(np.arange(n),x_dagger,alpha=0)
    plt.plot(np.arange(n),x)
    plt.plot(np.arange(n),x_dagger)
    plt.xlabel("x and x_dagger")
    plt.ylabel("y")
    plt.show()

def plot_x_dagger_delta(n):
    np.random.seed(114514)
    (K,x)=midpoint_quadrature_rule(n,0,1,0,1)
    (u,s,v_T)=np.linalg.svd(K)
    s=s*np.eye(n,dtype=np.float32)
    y=K@x
    delta=np.random.rand(n,1)*0.01
    y_delta=y+delta
    x_dagger_delta=v_T.T@np.linalg.pinv(s)@u.T@y_delta
    fig , ax = plt.subplots()
    ax.scatter(np.arange(n),x,alpha=0)
    ax.scatter(np.arange(n),x_dagger_delta,alpha=0)
    plt.plot(np.arange(n),x)
    plt.plot(np.arange(n),x_dagger_delta)
    plt.xlabel("x and x_dagger_delta")
    plt.ylabel("y")
    plt.show()

def plot_x_alpha_delta(n,a,flag):
    np.random.seed(114514)
    alpha=a
    (K,x)=midpoint_quadrature_rule(n,0,1,0,1)
    (u,s,v_T)=np.linalg.svd(K)
    s=s*np.eye(n,dtype=np.float32)
    y=K@x
    delta=np.random.rand(n,1)*0.01
    y_delta=y+delta
    F_alpha=s**2@np.linalg.pinv(s**2+np.eye(n,dtype=np.float32)*alpha)
    Psi_alpha=v_T.T@F_alpha@np.linalg.pinv(s)@u.T@y_delta
    if flag:
        fig , ax = plt.subplots()
        ax.scatter(np.arange(n),x,alpha=0)
        ax.scatter(np.arange(n),Psi_alpha,alpha=0)
        plt.plot(np.arange(n),x)
        plt.plot(np.arange(n),Psi_alpha)
        plt.xlabel("x and x_alpha_delta")
        plt.ylabel("y")
        plt.show()
    return x,Psi_alpha

def get_best_alpha(times):
    residual_error=1000
    min_n=0
    for n in np.logspace(0,1,times):
        x_gt,x_alpha_delta=plot_x_alpha_delta(100,n-1,0)
        tmp=np.linalg.norm(x_gt-x_alpha_delta)/np.linalg.norm(x_gt)
        if residual_error>tmp:
            residual_error=tmp
            min_n=n-1
    print(residual_error,min_n)
    plot_x_alpha_delta(100,min_n,1)

if __name__=='__main__':
    plot_x_alpha_delta(100,0.001,1)