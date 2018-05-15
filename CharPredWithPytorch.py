import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def predToOneHot(pred):

    #idx = np.argmax(pred[0])
    sortIdx = np.argsort(pred[0])
    reverseSortIdx = sortIdx[::-1]
    randomInt = randint(0,2)
    idx = reverseSortIdx[randomInt]
    oneHot = np.zeros_like(pred)
    oneHot[0,idx] = 1

    return oneHot

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
    vect[index] = 1
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

    def forward(self, inputs, train=True, steps=0):

        if train or steps == 0:
            steps = inputs.size()[0]

        outputs = torch.zeros(steps, inputs.size()[1], vectSize)

        hx = torch.zeros(inputs.size()[1], self.hidden_size).cuda()
        cx = torch.zeros(inputs.size()[1], self.hidden_size).cuda()

        for i in range(steps):
            if train or i == 0:
                input = inputs[i]
            else:
                input = output

            input = self.inp(input)
            hx, cx = self.rnn(input, (hx, cx))
            output = self.out(hx)

            outputs[i] = output

        return outputs, (hx,cx)

model = SimpleRNN(nbNeuron).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================================================================

# how many char in the dataset and how separate to make batche size of 25 x 25
#4366250 / 25 -> 174650, 174650 / 25 -> 6986

loopStep = 25 + batch_size
nbEpoch = 1
nbChar = 50000# len(strList) - loopStep - 1
lossList = []
stepCounter = 1
totalStep = int((nbChar / loopStep) * nbEpoch)

for k in range(nbEpoch):

    for i in range(0, nbChar, loopStep):

        matrix = []

        for j in range(25):

            idx = i+j
            matrix += [vectList[idx:idx+batch_size]]

        matrix = np.array(matrix)

        xData = matrix[0:24]
        xData = torch.FloatTensor(xData).cuda()

        yData = matrix[1:25]
        yData = np.argmax(yData, axis=2)
        yData = torch.LongTensor(yData)
        yData = yData.view(24*batch_size).cuda()

        outputs, hidden = model(xData, True)

        optimizer.zero_grad()
        loss = criterion(outputs.view(-1, 77).cuda(), yData)
        loss.backward()
        optimizer.step()

        currentLoss = loss.data.cpu().numpy()
        lossList += [currentLoss]

        if ((stepCounter - 1) % 500 == 0):
            print("Step : " + str(stepCounter) + " / " + str(totalStep) + ", Current Loss : " + str(currentLoss))
        stepCounter += 1



testName = "RnnPytorchImplementation"

plt.plot(lossList)
plt.savefig("./PerformanceData/" + testName + "_LossGraph.png")
plt.show()

print("---------------------------------------------")

batch_size = 1
strPred = ""
charsToTest = ["S","T","A"]

for char in charsToTest:

    allPred = []

    with torch.no_grad():
        charInput = torch.FloatTensor(np.array([[charToVect[char]]])).cuda()
        outputs, hidden = model(charInput, False, 500)

    strPred += char
    predictions = outputs.cpu().numpy()

    for vect in predictions:

        bestOptions = vect[0].argsort()[::-1]
        index = int(vect[0][bestOptions[np.random.randint(0,3)]])
        strPred += allChar[index]

    strPred += "\n\n\n"

text_file = open("./PerformanceData/" + testName + "_GenerateText.txt", "w")
text_file.write(strPred)
text_file.close()
print(strPred)
