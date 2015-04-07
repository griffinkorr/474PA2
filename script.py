import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
                

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD


    #Define Array Dimensions
    numN = X.shape[0]
    numD = X.shape[1]

    #Find number of k's
    numK = 0
    foundK = False
    ks_found = np.array([])
    ks_found = np.zeros(numN)
    for i in range(numN):
        for j in range(numN):
            if y[i] == ks_found[j]:
                 foundK = True
        if foundK == False:
                 numK = numK + 1
                 ks_found[numK] = y[i]
        foundK = False
   
    #Create Average Matrix
    means = np.zeros((numD,numK))

    countKs = np.zeros(numK)
    for n in range(numN):
        kLabel = int(y[n]) - 1
        for d in range(numD):
            means[d][kLabel] = means[d][kLabel] + X[n][d]
        countKs[kLabel] = countKs[kLabel] + 1

    for d in range(numD):
        for k in range(numK):
            means[d][k] = means[d][k] / countKs[k]

    #Create Covmariance Matrix
    covmats = np.zeros(((numK,numD,numD)))
    for k in range(numK):
        for d in range(numD):
            for n in range(numN):
                if int(y[n]-1) == k:
                   covmats[k][d][d] = covmats[k][d][d] + ((X[n][d] - means[d][k]) * (X[n][d] - means[d][k]))
            covmats[k][d][d] = covmats[k][d][d] / countKs[k]
    
    print(means)
    print(covmats)
                
    return means,covmats

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 

    # IMPLEMENT THIS METHOD
    numN = X.shape[0]
    numD = X.shape[1]

    #Find number of k's
    numK = 0
    foundK = False
    ks_found = np.array([])
    ks_found = np.zeros(numN)
    for i in range(numN):
        for j in range(numN):
            if y[i] == ks_found[j]:
                 foundK = True
        if foundK == False:
                 numK = numK + 1
                 ks_found[numK] = y[i]
        foundK = False
   
    #Create Average Matrix
    means = np.zeros((numD,numK))

    countKs = np.zeros(numK)
    for n in range(numN):
        kLabel = int(y[n]) - 1
        for d in range(numD):
            means[d][kLabel] = means[d][kLabel] + X[n][d]
        countKs[kLabel] = countKs[kLabel] + 1

    for d in range(numD):
        for k in range(numK):
            means[d][k] = means[d][k] / countKs[k]

    #Create Covmariance Matrix
    covmat = np.zeros(((numD,numD)))
    for k in range(numK):
        for d in range(numD):
            for n in range(numN):
                   if k+1 == y[n]:
                        covmat[d][d] = covmat[d][d] + ((X[n][d] - means[d][k]) * (X[n][d] - means[d][k]))
            covmat[d][d] = covmat[d][d] / countKs[k]
                    
    return means,covmat

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    #Define Array Dimensions
    numN = Xtest.shape[0]
    numD = Xtest.shape[1]
    numK = means.shape[1]

    print(covmat)

    acc = 0
    for n in range(numN):
        ks = np.zeros(numK)
        for k in range(numK):

            mdet = np.linalg.det(covmat)
            #print("Mdet")
            #print(mdet)

            if mdet != 0:
               mew = np.zeros(numD)
               meansT = means.T
               mew = meansT[k]
               #print("Mew")
               #print(mew)

               firstHalf = 1 / ((2*pi)**(numD/2)*(mdet)**(1/2))
               secondHalf = firstHalf * np.e**((-1*(Xtest[n]-mew).T * covmat**-1 * (Xtest[n]-mew))/2)
               ks[k] = np.linalg.det(secondHalf)

        index = np.argmax(ks)

        if int(ytest[n]) == index+1:
           acc = acc+1

    acc = acc/n


    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD

   
    #Define Array Dimensions
    numN = Xtest.shape[0]
    numD = Xtest.shape[1]
    numK = means.shape[1]

    #print(covmats)

    acc = 0
    for n in range(numN):
        ks = np.zeros(numK)
        for k in range(numK):

            covmat = np.zeros((numD,numD))
            covmat = covmats[k]
            mdet = np.linalg.det(covmat)

            if mdet != 0:

                mew = np.zeros(numD)
                meansT = means.T
                mew = meansT[k]
 
                firstHalf = 1 / ((2*pi)**(numD/2)*(mdet)**(1/2))
                secondHalf = firstHalf * np.e**((-1*(Xtest[n]-mew).T * covmat**-1 * (Xtest[n]-mew))/2)
                ks[k] = np.linalg.det(secondHalf)
           
        index = np.argmax(ks) 

        if int(ytest[n]) == index+1:
           acc = acc+1

    acc = acc/n


    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 
            
    
    w = (((X.T).dot(X))**-1).dot((X.T.dot(y)))
    
    def minimizeFunc1(w,X,y):

    #Declare Array Dimension Sizes
        numN = X.shape[0]
        sum = 0

        for n in range(numN):
            sum = sum + (y[n] - w.T.dot(X[n]))**2

        #print(sum/2)
        return sum/2

    args = (X,y)
    opts = {'maxiter' :50, 'disp' :True}
    #print(w)

    nn_params = minimize(minimizeFunc1, w, jac=False, args=args, method='Powell', options=opts)                          
    print(nn_params)
    print(w)
    return w

def learnRidgeERegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD

    #def minimizeFunc2(w,Xtest,ytest):

    #Declare Array Dimension Sizes
    numN = Xtest.shape[0]
    sum = 0    

    for n in range(numN):
        sum = sum + (ytest[n] - w.T.dot(Xtest[n]))**2

    rmse = (1/numN) * sqrt(sum) 

    return rmse

    #args = (Xtest,ytest)
    #opts = {'maxiter' :50}
    
    #nn_params = minimize(minimizeFunc2, w, jac=False, args=args, method='CG', options=opts)

    #print(nn_params)


def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 1.0, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
lambdas = np.linspace(0, 1.0, num=k)
k = 21
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 50}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    rmses4[i] = testOLERegression(w_l.x,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmax(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lamda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend('No Regularization','Regularization')
