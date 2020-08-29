#perceptron llearning algorithm for the and gate

import numpy as np     #mathematical stuffs

class Perceptron(object):      #(object ) means extends object 
    
#def __init__(self,no_of_inputs,threshold=10,learning_rate=0.1):  #2 mate
     
    def __init__(self,no_of_inputs,threshold=100,learning_rate=0.1):
        
        self.threshold=threshold
        self.learning_rate=learning_rate
        self.weights=np.zeros(no_of_inputs+1)   # + 1 for bias, [0,0,0,0,0] values vado default 
                                               # array create thayo
        #starting ma weights 0 lai lidha pachhi increase karsu.
    
    def predict(self,inputs): 
        
        summation = np.dot(inputs,self.weights[1:]) + self.weights[0]      # (w * x )+ b
        #print("summation :=>",summation)
        if(summation>0):
            activation=1
        else:
            activation=0
            
        return activation    

    def train(self,training_inputs,labels):
        #count=0
        for _ in range(self.threshold):
            #count+=1
            for inputs,label in zip(training_inputs,labels):   

                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] +=self.learning_rate * (label-prediction) * 1
            
            #print("1 to n = ",self.weights[1:])
            print("bias weight = ",self.weights[0])
    
                
        #print(count)

#learning-rate  => weights ketal thi increase thavu joiye.

'''
import numpy

numpy.zeros(5)
Out[72]: array([0., 0., 0., 0., 0.])


----------------------------------
listl=list(zip(a,b))

listl
Out[86]: [(2, 4), (3, 9), (4, 16), (5, 25), (6, 36)]

for i,j in zip(a,b): 
    print(i,j)
    
2 4
3 9
4 16
5 25
6 36

for i,j in zip(a,b): 
    print(i)
    
2
3
4
5
6

'''