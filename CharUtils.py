import numpy as np

fileName = "./shakespeare-plays/alllines.txt"

file = open(fileName,"r")
print(file)

strFile = file.read()
strFile = strFile.replace('"', '')
strList = list(strFile)

nbChar = 4366288

#4366275 / 25 -> 174651
#4366250 / 25 -> 174650, 174650 / 10 -> 17465
#4366250 / 25 -> 174650, 174650 / 25 -> 6986



allChar = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -,.?!:()[]'$\t\n"
index = 0
charToVect = {}
vectSize = len(allChar)

for char in allChar:

    vect = np.zeros(vectSize)
    vect[index] = 1
    charToVect[char] = vect
    index += 1

nbRemaining = 0
for char in strList:

    if (not (char in allChar)):
        print(char)
        nbRemaining += 1

print(nbRemaining)


vectList = []
for char in strList:
    vectList += [charToVect[char]]

matrix = []

for j in range(3):

    idx = (j*3)
    matrix += [vectList[idx:idx+3]]

print(matrix)