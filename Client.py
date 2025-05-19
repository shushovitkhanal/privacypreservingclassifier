'''
Represents a client that needs email to be evaluated by
an encrypted machine learning model before the email
is sent/recieved

Code Inspired by the examples given in the Python Paillier Library github repository by
the creators of the cryptosystem
Source: https://github.com/data61/python-paillier/blob/master/examples/logistic_regression_encrypted_model.py
'''


class Client:

    # Initialize method
    def __init__(self, public_key):
        self.public_key = public_key

    '''
    Returns an encrypted score based on how similar it is
    to the machine learning model
    
    @param weight - weight parameter passed through the encrypted ML model
    @param intercept - intercept paramter passed through the encrypted ML model
    @param X - Vectorized data to be tested
    '''
    def encrypted_predict(self, weight, intercept, X):
        scores = []
        # iterating through each row of the data
        for i in range(X.shape[0]):
            score = intercept
            for j in range(X.shape[1]):
                score += X[i, j] * weight[j]
            scores.append(score)
        return scores
