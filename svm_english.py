# -*- coding: utf-8 -*-
#Made available at http://www.mariofilho.com

import csv
from math import sqrt
from random import uniform, shuffle,seed, choice, randint


#PEGASOS


class SVMPegasos():

    def __init__(self,lmbd,D):
        self.lmbd = lmbd # regularization coefficient
        self.D = D + 1 # feature dimension plus bias
        self.w = [0.] * self.D # model weights
    
    # Function that attributes the class of the example according to the side of hyperplane it falls in
    def sign(self, x):
        return -1. if x <= 0 else 1.

    #Hinge Loss, function to be minimized
    def hinge_loss(self,target,y):
        return max(0, 1 - target*y)

    #Generator to load and feed the data
    def data(self,test=False):
        
        if test:
            with open('test.csv','r') as f:
                samples = f.readlines()
                
                for t,row in enumerate(samples):
                    
                    row = row.replace('\n','')
                    row = row.split(',')
                    
                    target = -1.
                    
                    if row[3] == '1':
                        target = 1.
                    del row[3]
                    
                    
                    x = [1.] + [float(c) for c in row] # bias + inputs
            
                    yield t, x,target
    
        else:
        
            with open('train.csv','r') as f:
                samples = f.readlines()
                shuffle(samples)
                
                for t,row in enumerate(samples):
            
                        
                    row = row.replace('\n','')
                    row = row.split(',')
                    
                    target = -1.
                        
                    if row[3] == '1':
                        target = 1.
                    del row[3]
                    
                    

                    x = [1.] + [float(c) for c in row] # bias + inputs

                    yield t, x,target




    # Trains in a single example
    def train(self,x,y,alpha):

        #if y is incorrect
        if y*self.predict(x) < 1:
        
            for i in xrange(len(x)):
                self.w[i] = (1. - alpha*self.lmbd)*self.w[i] + alpha*y*x[i]

        else:
            for i in xrange(len(x)):
               self.w[i] = (1. - alpha*self.lmbd)*self.w[i]


    #Dot product between weights and inputs
    def predict(self,x):
        wTx = 0.
        for i in xrange(len(x)):
            
            wTx += self.w[i]*x[i]

        return wTx



    #Trains SVM in the training set and estimates errors in validation
    def fit(self):
        

        test_count = 0.
        
        tn = 0.
        tp = 0.
        
        total_positive = 0.
        total_negative = 0.

        accuracy = 0.
        loss = 0.
        
        
        #last = 0
        for t, x,target in self.data(test=False):
            
            #if target == last:
            #   continue
            
            alpha = 1./(self.lmbd*(t+1.))
            self.train(x,target,alpha)
            #last = target
    
        for t,x,target in self.data(test=True):
            
            pred = self.predict(x)
            loss += self.hinge_loss(target,pred)
            
            pred = self.sign(pred)
            
            
            if target == 1:
                total_positive += 1.
            else:
                total_negative += 1.
            
            if pred == target:
                accuracy += 1.
                if pred == 1:
                    tp += 1.
                else:
                    tn += 1.



            
        loss = loss / (total_positive+total_negative)


        print 'Loss', loss, '\nTrue Negatives', tn/total_negative * 100, '%', '\nTrue Positives', tp/total_positive * 100, '%', '\nAccuracy', accuracy/(total_positive+total_negative) * 100, '%', '\n'

        return loss,tp/total_positive,tn/total_negative


loss_list = []
tp_list = []
tn_list = []

for i in range(100):
    
    print '\nSeed',i
    
    seed(i)
    
    svm = SVMPegasos(1e-1,3)

    l,tp,tn = svm.fit()

    loss_list.append(l)
    tp_list.append(tp)
    tn_list.append(tn)

print 'Mean Loss', sum(loss_list)/len(loss_list)
print 'True Positives', sum(tp_list)/len(tp_list) * 100, '%'
print 'True Negatives',sum(tn_list)/len(tn_list) * 100, '%'





