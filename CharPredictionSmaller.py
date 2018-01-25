from NetUtils import *
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
vectSize = len(allChar)

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
num_steps = 24
nbNeuron = 128

x = tf.placeholder(tf.float32, [batch_size, num_steps, vectSize], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size, num_steps, vectSize], name='labels_placeholder')

# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
rnn_inputs = tf.unstack(x, num=num_steps, axis=1)

def rnn(x, vectSize, nbNeuron, cellState, reuse):

    with tf.variable_scope("rnn", reuse=reuse):

        out = fullyLayer("1",x,vectSize,nbNeuron)

        out = fullyLayerNoRelu("2", out, nbNeuron, nbNeuron)
        out = tf.tanh(out)

        out, state = customCell_1("1", out, nbNeuron, cellState)

        pred = fullyLayerNoRelu("3", out, nbNeuron, vectSize)

    return pred, state

toReuse = False
init_state = tf.placeholder(tf.float32, shape=[None, nbNeuron])
state = init_state
logits = []

for rnn_input in rnn_inputs:
    pred, state = rnn(rnn_input, vectSize, nbNeuron, state, toReuse)
    logits.append(pred)
    toReuse = True

final_state = state

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
lr = 0.3
lrp = tf.placeholder(tf.float32, [])
losses = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(lrp).minimize(total_loss)
#AdagradOptimizer
#AdamOptimizer

x2 = tf.placeholder(tf.float32, shape=[None, vectSize])
pCellState = tf.placeholder(tf.float32, shape=[None, nbNeuron])

finalPred, cellState = rnn(x2, vectSize, nbNeuron, pCellState, True)
softmax = tf.nn.softmax(finalPred)


vars = tf.trainable_variables()
for v in vars:
    print(v.name)

clipVal = 2
clip = [p.assign(tf.clip_by_value(p, -clipVal, clipVal)) for p in vars]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ==============================================================================

# how many char in the dataset and how separate to make batche size of 25 x 25
#4366250 / 25 -> 174650, 174650 / 25 -> 6986

loopStep = 25 * batch_size
nbEpoch = 10
nbChar = len(strList) - loopStep - 1
lossList = []
stepCounter = 1
totalStep = int((nbChar / 505) * nbEpoch)

for k in range(nbEpoch):

    currentState = np.zeros((batch_size,nbNeuron))

    for i in range(0, nbChar, 505):

        matrix = []

        for j in range(batch_size):

            idx = i+(j*25)
            matrix += [vectList[idx:idx+25]]

        matrix = np.array(matrix)
        xData = matrix[:,0:24]
        yData = matrix[:,1:25]

        currentState, loss, _, _ = sess.run([final_state, total_loss, train_step, clip], feed_dict={x: xData, y: yData, init_state: currentState, lrp:lr})
        lossList += [loss]

        if ((stepCounter - 1) % 500 == 0):
            print("Step : " + str(stepCounter) + " / " + str(totalStep) + ", Current Loss : " + str(lossList[(len(lossList) - 1)]))
        stepCounter += 1



testName = "CustomCell1Adagrad_Lr-" + str(lr) + "_Ep-" + str(nbEpoch) + "_Hs-" + str(nbNeuron) + "_Cp-" + str(clipVal) + "_Bs-" + str(batch_size)

plt.plot(lossList)
plt.savefig("./PerformanceData/" + testName + "_LossGraph.png")
plt.show()

print("---------------------------------------------")

batch_size = 1
strPred = ""
charsToTest = ["So shaken as we are, so wan with care,","Within this hour it will be dinner-time:","You sheep, and I pasture: shall that finish the jest?","T","A"]

for chars in charsToTest:

    currentState = np.zeros((batch_size, nbNeuron))

    allPred = []

    for char in chars:

        charPred = np.array([charToVect[char]])
        allPred += [charPred[0]]
        currentState, charPred = sess.run([cellState, softmax],feed_dict={x2:charPred,pCellState:currentState})

    charPred = predToOneHot(charPred)
    allPred += [charPred[0]]

    for i in range(500):

        currentState, charPred = sess.run([cellState, softmax], feed_dict={x2: charPred, pCellState:currentState})
        charPred = predToOneHot(charPred)
        allPred += [charPred[0]]

    for vect in allPred:

        index = np.argmax(vect)
        strPred += allChar[index]

    strPred += "\n\n\n"

text_file = open("./PerformanceData/" + testName + "_GenerateText.txt", "w")
text_file.write(strPred)
text_file.close()
print(strPred)
