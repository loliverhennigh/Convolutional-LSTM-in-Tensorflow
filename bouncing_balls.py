
#######################################################
#
# This code was taken from Ilya Sutskever project on 
# Recurrent Temporal Restricted Boltzmann Machines
# The original source can be found
# http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
# There have been a few modifications to this code
# including adding gravity and dampening
#
#######################################################


from pylab import *

import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('friction', False,
                            """whether there is friction in the system""")
tf.app.flags.DEFINE_integer('num_balls', 2,
                            """num of balls in the simulation""")


def norm(x): return sqrt((x**2).sum())
def sigmoid(x):        return 1./(1.+exp(-x))


SIZE=10

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r==None: r=array([4.0]*n)
    if m==None: m=array([1]*n)
    # r is to be rather small.
    X=zeros((T, n, 2), dtype='float')
    V = zeros((T, n, 2), dtype='float')
    v = randn(n,2)
    v = (v / norm(v)*.5)*1.0
    good_config=False
    while not good_config:
        x = 2+rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<0:      good_config=False
                if x[i][z]+r[i]>SIZE:     good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False
                    
    
    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]
            V[t,i]=v[i]
            
        for mu in range(int(1/eps)):


            for i in range(n):
                #x[i]+=eps*v[i]
                x[i]+=.5*v[i]
            
            # gravity and drag
            if FLAGS.friction: 
                for i in range(n):
                    if (x[i][1]+r[i] < SIZE): v[i,1]+=.003
                    v[i]+=-(.005*v[i])
            

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: v[i][z]=-abs(v[i][z]) # want negative
            for i in range(n):
                    for j in range(i):
                        if norm(x[i]-x[j])<r[i]+r[j]:
                            #if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                            #  if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                              # the bouncing off part:
                            w    = x[i]-x[j]
                            w    = w / norm(w)
  
                            v_i  = dot(w.transpose(),v[i])
                            v_j  = dot(w.transpose(),v[j])
  
                            new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                         
                            v[i]+= w*(new_v_i - v_i)
                            v[j]+= w*(new_v_j - v_j)
  


            '''
            if flip: 
                flip = False
                for i in range(n):
                    for j in range(i):
                        if norm(x[i]-x[j])<r[i]+r[j]:
                            #if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                            #  if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                              # the bouncing off part:
                            w    = x[i]-x[j]
                            w    = w / norm(w)
  
                            v_i  = dot(w.transpose(),v[i])
                            v_j  = dot(w.transpose(),v[j])
  
                            new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                         
                            v[i]+= w*(new_v_i - v_i)
                            v[j]+= w*(new_v_j - v_j)
  
            else:
                flip = True
                for i in range(n):
                    for j in range(i):
                        if norm(x[(n-1)-i]-x[(n-1)-j])<r[(n-1)-i]+r[(n-1)-j]:
                            #if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                            #  if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                                # the bouncing off part:
                            w    = x[(n-1)-i]-x[(n-1)-j]
                            w    = w / norm(w)
    
                            v_i  = dot(w.transpose(),v[(n-1)-i])
                            v_j  = dot(w.transpose(),v[(n-1)-j])
    
                            new_v_i, new_v_j = new_speeds(m[(n-1)-i], m[(n-1)-j], v_i, v_j)
                         
                            v[(n-1)-i]+= w*(new_v_i - v_i)
                            v[(n-1)-j]+= w*(new_v_j - v_j)
    
            '''
    return X, V

def ar(x,y,z):
    return z/2+arange(x,y,z,dtype='float')

def matricize(X,V,res,r=None):

    T, n= shape(X)[0:2]
    if r==None: r=array([4.0]*n)

    A=zeros((T,res,res, 3), dtype='float')
    
    [I, J]=meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t, :, :, 1] += exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )
            A[t, :, :, 0] += 1.0 * (V[t,i,0] + .5) * exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )
            A[t, :, :, 2] += 1.0 * (V[t,i,1] + .5) * exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )
            
        A[t,:,:,0][A[t,:,:,0]>1]=1
        A[t,:,:,1][A[t,:,:,1]>1]=1
        A[t,:,:,2][A[t,:,:,2]>1]=1
    return A

def bounce_mat(res, n=2, T=128, r =None):
    if r==None: r=array([1.2]*n)
    x = bounce_n(T,n,r);
    A = matricize(x,res,r)
    return A

def bounce_vec(res, n=2, T=128, r =None, m =None):
    if r==None: r=array([1.2]*n)
    x,v = bounce_n(T,n,r,m);
    V = matricize(x,v,res,r)
    return V
    
def show_single_V(V):
    res = int(sqrt(shape(V)[0]))
    show(V.reshape(res, res))

def show_V(V):
    T   = len(V)
    res = int(sqrt(shape(V)[1]))
    for t in range(T):
        show(V[t].reshape(res, res))    

def unsigmoid(x): return log (x) - log (1-x)

def show_A(A):
    T = len(A)
    for t in range(T):
        show(A[t])


