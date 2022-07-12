# Logistic Regression in the flavor of the Gradient Descent 
import numpy as np 

class LogisticRegression(): 

    def __init__(self, X, alpha = 0.3, batch_size = None, iteration = None) -> None:

        """
        w: Randomly initialised weights on each feature 
        b: Randomly intitalised bias 
        alpha: leanrning rate, default 0.3
        """
        
        self.w = np.random.rand(X.shape[1],1)
        self.b = np.random.rand(1,1)
        self.alpha = alpha 
        self.batch_size = batch_size 
        self.iteration = iteration 

    def forward(self,X,w,b) -> float: 

        """
        X Numpy array = batch size * m 

        w Numpy array = m * 1 (random initialise)

        b Numpuy array = 1 * 1 (random initialise)
        """
        # liner function 
        Z = np.dot(X,w) + b 

        # activation 
        Y_hat = 1/(1 + np.exp(-Z))

        return Y_hat

    def backward(self, X, Y, Y_hat) -> float: 

        """
        X: (batch size, m)
        Y: (batch size ,1)
        Y_hat: (batch size * 1 )
        """
        # dZ = (batch size,1)
        dZ = Y-Y_hat 

        # dW = (m,1)
        dw = np.dot((np.transpose(X)), dZ)/X.shape[0]

        # db = (1,1)
        db = np.sum(dZ)/X.shape[0] 
        
        # GDS 
        self.w -= self.alpha * dw 
        self.b -= self.alpha * db 

        return self.w, self.b 

    def fit(self, model, X, Y) -> float: 

        """
        Wraper function that conduct ierative training across each batches 

        X: The feature  (N, m)

        Y: The label (N,1)
        """
        while i <= iteratio
            
            Y_hat_batch = model.forward(X_batch, self.w, self.b) 

            self.w, self.b = model.backward(X_batch, Y_batch, Y_hat_batch)

            print(self.w, self.b)

        return self.w, self.b

    def predict_probability(self,X) -> float:  

        """
        X Numpy array = batch size * m 
        """ 

        linear = np.dot(X , self.w) + self.b 

        return  1/(1+np.exp(-linear))