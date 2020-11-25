from joblib import Parallel, delayed
import multiprocessing
import json
import csv
import time
import re
from random import choice
from pprint import pprint
from fileParsing import *
from metrics import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def dataAccess():
    folder = createTxt()
    files = os.listdir(folder)
    documents = []
    for f in files:
        if ".txt" in f:
            documents.append(f)
    numWords = lenCheck(folder, documents)
    values = csvCreation(numWords, documents, folder)
    values = np.asarray(values)
    return values, documents


def modelCreation(trainX, trainY):
    model = Sequential()
    model.add(Dense(15, input_dim=len(trainX[0]), activation='selu'))
    model.add(Dense(10, activation='selu'))
    model.add(Dense(5, activation='selu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=200,
              batch_size=5, verbose=0, shuffle=True)
    return model


def statiscalResult():
    return


def dataPreperation(notDoc, data):
    np.random.seed(int(time.time()))
    d = np.empty((0, len(data[0][0]) + 1))
    for i in range(0, len(data)):
        if i == notDoc:
            docVal = np.zeros((len(data[i]), 1))
            row = np.hstack((data[i], docVal))
            np.random.shuffle(row)
            inputDim = (len(row[0]))
            testX = row[:, 0:inputDim - 1]
            testY = row[:, inputDim - 1]
        else:
            docVal = np.zeros((len(data[i]), 1))
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
        r = Parallel(n_jobs=num_cores)(delayed(runSimulation)
                                       (data, documents, i)for j in range(0, tests))
        m = []
        for tup in r:
            m.append(tup[2])

        results.append((documents[i], np.median(m), m))
    return results


def predictResult(results):
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

# currently not applied. This is a pre-run to calculate chunk size, ended up not being accurate enough and subsitited for a distribution


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
    print((np.max(val) - np.min(val)))
    return ((np.max(val) - np.min(val)))


def runSimulation(data, documents, num):
    results = []
    med = []
    mins = []
    maxs = []
    trainX, trainY, testX, testY = dataPreperation(num, data)
    model = None
    model = modelCreation(trainX, trainY)
    res = model.evaluate(testX, testY, verbose=0)
    val = model.predict(testX)
    med.append(val)
    mins.append(np.min(val))
    maxs.append(np.max(val))
    return (documents[num], np.median(med), med)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    a = time.time()
    rTot = []
    data, documents = dataAccess()
    r = []
    v = []
    medians = []
    for doc in documents:
        medians.append([])
    for i in range(0, 10):
        print("Rotataion", i+1)
        results = runModelTests(5, data, documents)
        r.append(results)
        for i in range(0, len(results)):
            for j in range(0, len(results[i][2])):
                medians[i].append(results[i][2][j])
    # pprint(r)
    print("\n\n\nTOTAL MEDIANS\n\n\n")
    for i in range(0, len(documents)):
        print(documents[i], np.median(medians[i]))
        v.append((documents[i], np.median(medians[i])))
    rTot.append(v)
    b = time.time()
    print(predictResult(v))
    print(str((b - a) / 60))
