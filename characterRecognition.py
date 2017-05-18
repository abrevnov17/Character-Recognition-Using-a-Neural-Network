#notes for next time: weights are super small, sigmoid is rounding to 1, and something weird is happening with normalization since they don't add up to 1

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math


#variable definitions

#step size (decreases after each iteration)
stepSize = 10000
decreaseRate = 0.97
epochCount=0

#defines number of hidden nodes in our one hidden layer
hiddenNodesNumber = 50

#epochs we want to run
epochs = 100


#w1 is a 784 x 50 matrix -> initializing it with random values between 0 and 2...may need to adjust this range later for increased efficiency

w1 = np.random.rand(784,hiddenNodesNumber)

min = 0
max = 0.005
w1 = (np.random.rand(784,hiddenNodesNumber) * (max - min) ) + min

#w2 is a 50 x 9 matrix -> initializing it with random values between 0 and 2...may need to adjust this range later for increased efficiency

w2 = np.random.rand(hiddenNodesNumber,10)

min = 0
max = 0.005
w2 = (np.random.rand(hiddenNodesNumber,10) * (max - min) ) + min

#function that loads in the MNIST database images

def loadMNIST(path, kind='train'):

    #images is an n x m dimensional array where n is the number of samples and m is the number of features
    #we unroll the 28 x 28 pixels for each images into one dimensional row vectors which represent the rows of our image array

    #labels contains the corresponding target variable (class labels) for the images

    labelsPath = os.path.join(path, '%s-labels-idx1-ubyte' %kind)

    imagesPath = os.path.join(path, '%s-images-idx3-ubyte' %kind)

    with open(labelsPath, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels=np.fromfile(lbpath, dtype=np.uint8)

    with open(imagesPath, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images=np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)

    return images, labels
            
x,y = loadMNIST('mnist', kind='train')

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoidDot(first,second):
    return 1.0/(1.0+2.7182818284590452**(-first.dot(second)))

def computePsi(hiddenLayerOutput,correctDigit):
    #passed in a row vector, row, with a symbol, w, inside of it

    #finding numerator
    
    numerator = sigmoidDot(hiddenLayerOutput,Matrix(w2[:,correctDigit]))

    #finding denominator

    denominator = 0

    for i in range(10):
        denominator = denominator + sigmoidDot(Matrix(w2[:,i]),hiddenLayerOutput)

    function = numerator/denominator
    
    return function
def computePsi2(output,a,correctDigit,rowIndex):
    #passed in a row vector, row, with a symbol, w, inside of it

    #finding numerator

    if correctDigit == rowIndex:
        numerator = sigmoidDot(output,a)
    else:
        numerator = sigmoidDot(Matrix(w2[:,correctDigit]),a)

    #finding denominator

    denominator = 0

    for k in range(10):
        if rowIndex == k:
            denominator = denominator + sigmoidDot(a,w2[:,k])

        else:
            denominator = denominator + sigmoidDot(a,output)

    function = numerator/denominator
    
    return function

def calculateGradient(a,inputVector,correctDigit):
    a=Matrix(a)
    w=Symbol("w")
    print("begun calculating gradient")

    gradientVector = []

    count = 0

    #getting all values for gradient of w1 pre-partial derivative 
    
    for weight in np.nditer(w1):
        indexInRow=count%784
        if inputVector[indexInRow]!=0:
            #first thing we need to do is get the row vector with the variable inside of it

            rowIndex = np.ceil(count/784)

            numpyRow = w1[:,rowIndex]

            row=Matrix(numpyRow)
            row = row.row_insert(indexInRow, Matrix([[w]]))
            row.row_del(indexInRow+1)  


            #we now have a row vector,x, of w1 with a w inside it at index given by count2. Note: we only have to do this once per iteration since, for other row vectors, we don't need it to contain a symbol

            #our next task is come up with an expression for our function that we need to take the partial derivative of in respect to w
            hiddenLayerOutput = Matrix([[0]])
            
            for q in range(hiddenNodesNumber):
                #using the sigmoid function 1/(1+e^(-t)) as our activation function
                if (q == rowIndex):
                    hiddenLayerOutput = hiddenLayerOutput.row_insert(q+1,Matrix([[sigmoidDot(row,inputVector)]]))

                else:
                    hiddenLayerOutput = hiddenLayerOutput.row_insert(q+1,Matrix([[sigmoidDot(w1[:,q],inputVector)]]))


            hiddenLayerOutput.row_del(0)
            
            outputFunction = computePsi(hiddenLayerOutput,correctDigit)
            gradientVector.append(outputFunction)

        else:
            gradientVector.append(0)
            
        count=count+1
        
    print("pre-derivative gradient for W1")



    #now that we have those values, we examine the w2 matrix

    count = 0      

    for weight in np.nditer(w2):

        indexInRow = count%hiddenNodesNumber

        #first thing we need to do is get the row vector with the variable inside of it
        if a[indexInRow]!=0:
            rowIndex = np.ceil(count/hiddenNodesNumber)


            numpyRow = w2[:,rowIndex]

            row=Matrix(numpyRow)
            row = row.row_insert(indexInRow, Matrix([[w]]))
            row.row_del(indexInRow+1)  


            #we now have a row vector,x, of w1 with a w inside it at index given by count2. Note: we only have to do this once per iteration since, for other row vectors, we don't need it to contain a symbol

            #our next task is come up with an expression for our function that we need to take the partial derivative of in respect to w

            
            outputFunction = computePsi2(row,a,correctDigit,rowIndex)
            gradientVector.append(outputFunction)
        else:
            gradientVector.append(0)
            
        count=count+1

    count=0
    print("pre-derivative gradient for W2")

    for element in gradientVector:
        vector=diff(element,w)
        vector=vector.subs(w,w1[count/hiddenNodesNumber,count%hiddenNodesNumber])
        gradientVector[count]=vector
        count = count+1
        if count >= hiddenNodesNumber*784:
            break
        
    count=0
    print("derivatives gradient W1")
    
    for element in gradientVector:
        vector=diff(element,w)
        vector=vector.subs(w,w2[count/10,count%10])
        gradientVector[count+hiddenNodesNumber*784]=vector
        count = count+1
        if count>=hiddenNodesNumber*10:
            break;

    print("derivatives gradient W2")
    print("full gradient calculation complete")

    
    return gradientVector
        
        

def loopThroughImages(x,y):
    global stepSize
    global w1
    global w2

    for i in range(10):

        for j in range(x.shape[1]):

            inputVector = x[y == i][j]
            inputVector = inputVector/float(255)
            #print(inputVector)

            #here we actually run an iteration of our algorithm on an inputVector

            #we are using 50 hidden nodes...consider testing on different numbers of hidden nodes

            #a array is an array of our hidden layer values

            a = np.zeros(hiddenNodesNumber)

            for q in range(hiddenNodesNumber):
                
                #using the sigmoid function 1/(1+e^(-t)) as our activation function
                a[q] = sigmoid(np.dot(w1[:,q],inputVector))
                

            #using our hidden layer values mapped onto [0,1] using our sigmoid activation function, we use a similar process to determine our output values

            #outputPercentages is an indexed, stochastic list that contains the probabilities of the given character being each digit (0,9)

            outputPercentages = np.zeros(10)

            #at first we do very similar steps to determining the values of a, the difference will come later when we scale the values of outputPercentages in such a way so that their sum is 1

            magnitudeOfOutputPercentages = 0
            
            for q in range (10):
                outputPercentages[q] = sigmoid(np.dot(w2[:,q],a))
                magnitudeOfOutputPercentages += outputPercentages[q]

            for q in range (10):
                outputPercentages[q] = outputPercentages[q]/magnitudeOfOutputPercentages

            error = 1- outputPercentages[i]
            print 'error',error
            

            #need to calculate gradients
            #note: i is the correct digit

            gradientVector = calculateGradient(a,inputVector,i)
            gradientVector = np.array(gradientVector)

            #divide each element by the step size
            gradientVector = gradientVector*stepSize

            #add each element in gradient vector to corresponding weights
            
            count=0
            print("multiplying gradient vector by step size")
    
            for element in gradientVector:
                w1[count/hiddenNodesNumber,count%hiddenNodesNumber] = w1[count/hiddenNodesNumber,count%hiddenNodesNumber]+element
                count = count+1
                if count >= hiddenNodesNumber*784:
                    break
                
            count=0
            print("updated W1 final")
            
            for element in gradientVector:
                w2[count/10,count%10] = w2[count/10,count%10] + element
                count = count+1
                if count>=hiddenNodesNumber*10:
                    break;

            print("updated W2 final")
            
            #now that we have updated all of our weights, we decrease the step size a bit
            #decreaseRate changes the rate at which the step size decreases
            print("new epoch beginning")
            print
            stepSize = decreaseRate*stepSize
            
           
                
print("training initialized")
print

loopCount=0


while loopCount<epochs:
    print 'PercentComplete',loopCount
    print
    loopThroughImages(x,y)
