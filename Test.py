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
nbNeuron = 128

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):

        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(vectSize, hidden_size)
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vectSize)

    def forward(self, input, memory, steps=0):

        input = self.inp(input)
        hx, cx = self.rnn(input, memory)
        output = self.out(hx)

        return output, (hx,cx)

model = SimpleRNN(nbNeuron).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================================

# how many char in the dataset and how separate to make batche size of 25 x 25
#4366250 / 25 -> 174650, 174650 / 25 -> 6986

seqLentgh = 25 # real length is 24, 25 for last label
nbEpoch = 5
lossList = []
decal = 1

start = time.time()

for k in range(nbEpoch):

    if (decal > 10):
        decal = 1
    decal = decal + 1
    loopStep = seqLentgh + (batch_size * decal)
    nbChar = len(strList) - loopStep - 1
    stepCounter = 0
    epochStep = int(nbChar / loopStep)

    for i in range(0, nbChar, loopStep):

        data = []

        for j in range(batch_size):

            idx = i+(j*decal)
            data += [vectList[idx:idx+seqLentgh]]

        data = np.array(data)

        fullLoss = torch.zeros(1).cuda()

        hx = torch.zeros(batch_size, nbNeuron).cuda()
        cx = torch.zeros(batch_size, nbNeuron).cuda()
        hidden = (hx,cx)

        for idx in range(seqLentgh-1):

            xData = torch.FloatTensor(data[:,idx]).cuda()
            outputs, hidden = model(xData, hidden)
            target = data[:,idx+1]
            target = torch.FloatTensor(target).cuda()
            loss = criterion(outputs, target)
            fullLoss = torch.add(fullLoss, loss)

        optimizer.zero_grad()
        fullLoss.backward()
        optimizer.step()

        currentLoss = fullLoss.data.cpu().numpy()[0]
        lossList += [currentLoss]

        if (stepCounter % 500 == 0):
            print("Epoch : " + str(k+1) + " / " + str(nbEpoch) + ", Step : " + str(stepCounter) + " / " + str(epochStep) + ", Current Loss : " + str(currentLoss))
        stepCounter += 1

    end = time.time()
    timeTillNow = end - start
    predictedRemainingTime = (timeTillNow / (k + 1)) * (nbEpoch - (k + 1))
    print("--------------------------------------------------------------------")
    print("Finished epoch : " + str(k))
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
        hx = torch.zeros(1, nbNeuron).cuda()
        cx = torch.zeros(1, nbNeuron).cuda()
        hidden = (hx, cx)

        for char in chars:
            charVect = [charToVect[char]]
            charVect = torch.FloatTensor(charVect).cuda()
            output, hidden = model(charVect, hidden)
            lastOutput = output
            predictions += [output[0]]

        for i in range(500):
            output, hidden = model(lastOutput, hidden)
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
