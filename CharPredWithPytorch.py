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

    def forward(self, inputs, steps=0):

        inputSize = inputs.size()[0]

        if inputSize <= 0:
            raise ValueError('No input given before generation')

        outputs = torch.zeros(inputSize+steps, inputs.size()[1], vectSize).cuda()

        hx = torch.zeros(inputs.size()[1], self.hidden_size).cuda()
        cx = torch.zeros(inputs.size()[1], self.hidden_size).cuda()

        for i in range(inputSize):

            input = self.inp(inputs[i])
            hx, cx = self.rnn(input, (hx, cx))
            output = self.out(hx)

            outputs[i] = output

        for i in range(steps):

            input = self.inp(output)
            hx, cx = self.rnn(input, (hx, cx))
            output = self.out(hx)

            outputs[inputSize+i] = output

        return outputs, (hx,cx)

model = SimpleRNN(nbNeuron).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================================

# how many char in the dataset and how separate to make batche size of 25 x 25
#4366250 / 25 -> 174650, 174650 / 25 -> 6986

seqLentgh = 25 # real length is 24, 25 for last label
nbEpoch = 10
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

        temp = []
        for idx in range(seqLentgh):
            temp += [data[:,idx]]

        data = np.array(temp)

        xData = data[0:24]
        xData = torch.FloatTensor(xData).cuda()

        yData = data[1:25]
        #yData = np.argmax(yData, axis=2)
        yData = torch.FloatTensor(yData).cuda()
        #yData = yData.view(24*batch_size).cuda()

        outputs, hidden = model(xData)

        optimizer.zero_grad()
        loss = criterion(outputs, yData)
        loss.backward()
        optimizer.step()

        currentLoss = loss.data.cpu().numpy()
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

    allPred = []
    charVect = []

    for char in chars:

        charVect += [np.array([charToVect[char]])]

    with torch.no_grad():
        charInput = torch.FloatTensor(np.array(charVect)).cuda()
        outputs, hidden = model(charInput, 500)

    strPred += char
    predictions = outputs.cpu().numpy()

    for vect in predictions:

        index = predToOneHot(vect[0])
        strPred += allChar[index]

    strPred += "\n\n\n"

text_file = open("./PerformanceData/" + testName + "_GenerateText.txt", "w")
text_file.write(strPred)
text_file.close()
print(strPred)
