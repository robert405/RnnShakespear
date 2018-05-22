import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import time

def predToOneHot(pred):

    #idx = np.argmax(pred[0])
    sortIdx = np.argsort(pred)
    reverseSortIdx = sortIdx[::-1]
    randomInt = randint(0,2)
    idx = reverseSortIdx[randomInt]

    return idx

def randSeq(length, seq):

    seqSize = len(seq)
    randomInt = randint(0,seqSize-length-1)
    randSeq = seq[randomInt : randomInt+length]

    return randSeq

# ==============================================================================
# fetch data

fileName = "./shakespeare-plays/alllines.txt"

file = open(fileName,"r")
strFile = file.read()
strFile = strFile.replace('"', '')
strList = list(strFile)
file.close()

# ==============================================================================
#create dictionary to map char to vector

allChar = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -,.?!:()[]'$\t\n"
index = 0
charToVect = {}
vectSize = len(allChar) # = 77

for char in allChar:

    vect = np.zeros(vectSize)
    vect[index] = 3
    charToVect[char] = vect
    index += 1

vectList = []
for char in strList:
    vectList += [charToVect[char]]

# ==============================================================================

batch_size = 50
nbNeuron = 512

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):

        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lin1 = nn.Linear(vectSize, hidden_size)
        self.relu1 = nn.ReLU()
        self.rnn1 = nn.LSTMCell(hidden_size, hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vectSize)

    def forward(self, xData, memory1, memory2):

        xData = self.lin1(xData)
        xData = self.relu1(xData)
        hx1, cx1 = self.rnn1(xData, memory1)
        hx2, cx2 = self.rnn2(hx1, memory2)
        output = self.out(hx2)

        return output, (hx1,cx1), (hx2,cx2)

model = SimpleRNN(nbNeuron).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================================

# how many char in the dataset and how separate to make batche size of 25 x 25
#4366250 / 25 -> 174650, 174650 / 25 -> 6986

seqLentgh = 25 # real length is 24, 25 for last label
nbIt = 1000000
lossList = []

start = time.time()

for i in range(nbIt):

    data = []

    for j in range(batch_size):
        data += [randSeq(seqLentgh, vectList)]

    data = np.array(data)

    fullLoss = torch.zeros(1).cuda()

    hx1 = torch.zeros(batch_size, nbNeuron).cuda()
    cx1 = torch.zeros(batch_size, nbNeuron).cuda()
    hidden1 = (hx1,cx1)

    hx2 = torch.zeros(batch_size, nbNeuron).cuda()
    cx2 = torch.zeros(batch_size, nbNeuron).cuda()
    hidden2 = (hx2, cx2)

    for idx in range(seqLentgh-1):

        xData = torch.FloatTensor(data[:,idx]).cuda()
        outputs, hidden1, hidden2 = model(xData, hidden1, hidden2)
        target = data[:,idx+1]
        target = torch.FloatTensor(target).cuda()
        loss = criterion(outputs, target)
        fullLoss = torch.add(fullLoss, loss)

    optimizer.zero_grad()
    fullLoss.backward()
    optimizer.step()

    currentLoss = fullLoss.data.cpu().numpy()[0]
    lossList += [currentLoss]

    if (i % 500 == 0):
        print("Step : " + str(i) + " / " + str(nbIt) + ", Current Loss : " + str(currentLoss))

    if (i % 5000 == 0):
        end = time.time()
        timeTillNow = end - start
        predictedRemainingTime = (timeTillNow / (i + 1)) * (nbIt - (i + 1))
        print("--------------------------------------------------------------------")
        print("Time to run since started (sec) : " + str(timeTillNow))
        print("Predicted remaining time (sec) : " + str(predictedRemainingTime))
        print("--------------------------------------------------------------------")


end = time.time()
print("Time to run in second : " + str(end - start))

testName = "RnnPytorchImplementation"

plt.plot(lossList)
plt.savefig("./PerformanceData/" + testName + "_LossGraph.png")
plt.show()

print("---------------------------------------------")

batch_size = 1
strPred = ""
charsToTest = ["So shaken as we are, so wan with care,","Within this hour it will be dinner-time:","You sheep, and I pasture: shall that finish the jest?", "S","T","A"]

for chars in charsToTest:

    predictions = []

    with torch.no_grad():

        lastOutput = torch.zeros(1,vectSize).cuda()
        hx1 = torch.zeros(batch_size, nbNeuron).cuda()
        cx1 = torch.zeros(batch_size, nbNeuron).cuda()
        hidden1 = (hx1, cx1)

        hx2 = torch.zeros(batch_size, nbNeuron).cuda()
        cx2 = torch.zeros(batch_size, nbNeuron).cuda()
        hidden2 = (hx2, cx2)

        for char in chars:
            charVect = [charToVect[char]]
            charVect = torch.FloatTensor(charVect).cuda()
            output, hidden1, hidden2 = model(charVect, hidden1, hidden2)
            lastOutput = output
            predictions += [output[0]]

        for i in range(500):
            output, hidden1, hidden2 = model(lastOutput, hidden1, hidden2)
            lastOutput = output
            predictions += [output[0]]

    strPred += ""

    for vect in predictions:

        npVect = vect.data.cpu().numpy()
        index = predToOneHot(npVect)
        strPred += allChar[index]

    strPred += "\n\n\n"

text_file = open("./PerformanceData/" + testName + "_GenerateText.txt", "w")
text_file.write(strPred)
text_file.close()
print(strPred)
