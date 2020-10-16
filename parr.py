import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from metrics import *
from fileParsing import *
from pprint import pprint
from random import choice
import re
import time
import csv
import json
import multiprocessing
from joblib import Parallel, delayed


def dataAccess():
    folder = createTxt()
    files = os.listdir(folder)
    documents = []
    for f in files:
        if ".txt" in f:
            documents.append(f)
    #numWords = preRun(documents,folder)
    numWords = lenCheck(folder, documents)
    print(numWords)
    values = csvCreation(numWords, documents, folder)
    #documents.sort(key=lambda f: int(re.sub('\D', '', f)))
    values = np.asarray(values)
    return values, documents

# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def modelCreation(trainX, trainY):
    model = Sequential()
    model.add(Dense(15, input_dim=len(trainX[0]), activation='selu'))
    model.add(Dense(10, activation='selu'))
    model.add(Dense(5, activation='selu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy', f1])
    model.fit(trainX, trainY, epochs=200,
              batch_size=5, verbose=0, shuffle=True)
    return model


def statiscalResult():
    return


def dataPreperation(notDoc, data):
    np.random.seed(int(time.time()))
    d = np.empty((0, len(data[0][0]) + 1))
    for i in range(0, len(data)):
        #d = np.append(d,np.array([]))
        if i == notDoc:
            docVal = np.zeros((len(data[i]), 1))
            # print(data[i],docVal)
            row = np.hstack((data[i], docVal))
            # print(row[0])
            np.random.shuffle(row)
            inputDim = (len(row[0]))
            testX = row[:, 0:inputDim - 1]
            testY = row[:, inputDim - 1]
            # #print("Row",row)
            # #for val in row:
            # d2 = np.vstack((d2,row))
        else:
            docVal = np.zeros((len(data[i]), 1))
            # print(data[i],docVal)
            row = np.hstack((data[i], docVal))
            d = np.vstack((d, row))

    forTrain = np.loadtxt(
        "Fingerprints/" + choice(os.listdir("Fingerprints")), delimiter=",")
    np.random.shuffle(forTrain)
    forTrain = forTrain[:len(d)]
    trainZeros = np.ones((len(forTrain), 1))
    forTrain = np.concatenate((forTrain, trainZeros), axis=1)
    d = np.concatenate((d, forTrain))
    np.random.shuffle(d)
    inputDim = (len(d[0]))
    trainX = d[:, 0:inputDim - 1]
    trainY = d[:, inputDim - 1]
    return trainX, trainY, testX, testY


def runModelTests(tests, data, documents):
    results = []
    num_cores = multiprocessing.cpu_count()
    for i in range(0, len(data)):
        print("Rotataion", i)
        r = Parallel(n_jobs=num_cores)(delayed(runSimulation)
                                       (data, documents, i)for j in range(0, tests))
        # for j in range(0, tests):
        m = []
        for tup in r:
            m.append(tup[2])

        results.append((documents[i], np.median(m), m))
        print((documents[i], np.median(m)))
    return results


def predictResult(results):
    # This uses the Results from multiple tests
    #print(results)
    results.sort(key=lambda x: x[1])
    diff = results[-1][1] - results[-2][1]
    if diff >= 0.15:
        if len(results) == 2:
            return "The Documents are Written By Different Authors"
        else:
            return "The Document " + str(results[-1][0]) + " Is not written by the Same Author"
    elif diff < 0.15 and diff > 0.10:
        return "Run Another Simulation to be Sure of Authorship - Currently All Written by the Same Author"
    else:
        return "All Documents Are written by the Same Author"

# if __name__ == "__main__":
#     wNum = 100
#     r = []
#     while wNum <=500:
#         f = open("Test Data 3 - Multi Word Test.txt","a+")
#         r.append(("NUMBER OF WORDS",wNum))
#         f.write("NUMBER OF WORDS"+str(wNum)+"\n")
#         for i in range(0, 10):
#             data, documents = dataAccess(wNum)
#             results = runModelTests(5, data, documents)
#             r.append(results)
#             for t in results:
#               f.write(' '.join(str(s) for s in t)+", ")
#             #print(results)
#             f.write("\n")
#         wNum = wNum+50
#         f.write("\n")
#         f.close()
#     pprint(r)

    # Test first yaer or hs documents against curent work


def preRun(documents, folder):
    stored = []
    num_cores = multiprocessing.cpu_count()
    for i in range(0, 6):
        check = []
        data = csvCreation((250 + i * 50), documents, folder)
        for j in range(0, len(data)):
            r = Parallel(n_jobs=num_cores)(delayed(runWordNumTest)
                                           (data, documents, j)for y in range(0, 2))
            check.append(r)
        stored.append(np.max(check))
        print(np.max(check))
    return 250 + stored.index(np.max(stored)) * 50


def runWordNumTest(data, documents, num):
    results = []
    med = []
    mins = []
    maxs = []
    trainX, trainY, testX, testY = dataPreperation(num, data)
    model = None
    model = modelCreation(trainX, trainY)
    val = model.predict(testX)
    mins.append(np.min(val))
    maxs.append(np.max(val))
    # doc.append((np.min(val),np.median(val),np.max(val)))
    # print(med)
    # results.append()
    print((np.max(val) - np.min(val)))
    return ((np.max(val) - np.min(val)))

# Main 2, recoded for word len checks


def runSimulation(data, documents, num):
    results = []
    med = []
    mins = []
    maxs = []
    trainX, trainY, testX, testY = dataPreperation(num, data)
    model = None
    model = modelCreation(trainX, trainY)
    res = model.evaluate(testX, testY, verbose=0)
    print(res)
    val = model.predict(testX)
    med.append(val)
    mins.append(np.min(val))
    maxs.append(np.max(val))
    # doc.append((np.min(val),np.median(val),np.max(val)))
    # print(med)
    # results.append()
    return (documents[num], np.median(med), med)

if __name__ == "__main__":
    a = time.time()
    rTot = []
    data, documents = dataAccess()
# for x in range(0,10):
    r = []
    v = []
    medians = []
    for doc in documents:
        medians.append([])
    for i in range(0, 20):
        results = runModelTests(5, data, documents)
        r.append(results)
        for i in range(0, len(results)):
            for j in range(0, len(results[i][2])):
                medians[i].append(results[i][2][j])
        print()
        print([res[0:2] for res in results])
    pprint(r)
    print("\n\n\nTOTAL MEDIANS\n\n\n")
    for i in range(0, len(documents)):
        print(documents[i], np.median(medians[i]))
        v.append((documents[i], np.median(medians[i])))
    rTot.append(v)
    b = time.time()
    print("All Results")
    # f = open("Metric test 1.txt", "a+")
    # for row in v:
    #     f.write(' '.join(str(s) for s in row) + ", ")
    #     pprint(row)
    #     f.write("\n")
    # f.write(str(predictResult(v))+ "\n")
    # f.close()
    print(predictResult(v))
    print(str((b - a) / 60))
