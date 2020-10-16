import operator
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getNumWords(data):
    wordCount = 0
    newData = []
    for word in data:
        if not word.isalnum():
            doubleCheck = False
            for letter in word:
                if letter.isalnum():
                    doubleCheck = True
                    break
            if doubleCheck:
                wordCount += 1
                newData.append(word.lower())
        else:
            wordCount += 1
            newData.append(word.lower())
    return newData, wordCount


def avgSentenceLength(data):
    sTotal = 0
    sNum = 0
    count = 0
    for word in data:
        if "." in word or "?" in word or "!" in word:
            sTotal += count
            sNum += 1
            count = 0
        else:
            count += 1
    try:
        return sTotal / sNum
    except:
        return 0


def quotePercentage(data, wordCount):
    qLen = 0.0
    for word in data:
        word = word.lower()
        if "quote(" in word:
            if word[-1] != ")":
                qLen += int(word[6:-2])
            else:
                # note: this can be hardcoded due to the common format of
                # quotes
                qLen += int(word[6:-1])
    return qLen / wordCount


def wordCounts(data):
    wCount = {}
    wCountOrdered = []
    for word in data:
        count = wCount.get(word, 0)
        wCount[word] = count + 1
    wCount = sorted(wCount.items(), key=operator.itemgetter(1))
    for val in reversed(wCount):
        wCountOrdered.append(val)
    return wCountOrdered


def zipfs(data):
    x = []
    y = []
    lastX = None
    r2Check = 0.0
    check = False
    for i in range(0, len(data)):
        v1 = math.log10(i + 1)
        v2 = math.log10(data[i][1])
        x.append(v2)
        y.append(v1)

        if lastX is not None and lastX == 0.0 and not check:
            r2Check = np.corrcoef(x, y)[0][1]**2
            check = True
        lastX = v2
    r2 = np.corrcoef(x, y)[0][1]**2
    return r2, r2Check, x, y


def yulesNumbers(data, numWords):
    cMax = data[0][1]
    s2 = 0
    for i in range(0, cMax + 1):
        count = 0
        for val in data:
            # print(data[1])
            if val[1] == i:
                count += 1
        s2 += count * (i / numWords)**2
    yulesK = 10**4 * ((-1 / numWords) + s2)
    yulesI = 1 / s2
    return yulesK, yulesI


def pCount(data):
    puncCount = 0
    for word in data:
        if ',' in word or "?" in word or "!" in word or "'" in word or ";" in word:
            puncCount += 1
        print
    return puncCount / len(data)


def contractionCount(data):
    cCount = 0
    for word in data:
        if "-" in word or "'" in word:
            cCount += 1
        print
    return cCount / len(data)


def csvCreation(wordLim, files, folder):
    # wordLim = 250
    ret = []
    values = []
    dFile = []
    for file in files:
        dFile = []
        f = open(folder + "/" + file)
        data = f.read()
        data = data.split()
        d, wordCount = getNumWords(data)
        num = round(wordCount / wordLim)
        for j in range(0, num + 1):
            count = wordLim
            end = False
            lower = j * wordLim
            if (j == num or num == 1):
                upper = -1
                end = True
            else:
                upper = j * wordLim + wordLim
                if upper > wordCount and (wordCount - (j * wordLim)) >= (wordLim * 0.9):
                    count = wordCount - (j * wordLim)
                    upper = -1
                    end = True
                elif (wordCount - (j * wordLim)) < (wordLim * 0.9):
                    break
            if not d[lower:upper]:
                print(lower, upper)
                continue
            wCount = wordCounts(d[lower:upper])
            yuleK, yuleI = yulesNumbers(wCount, count)
            qPen = quotePercentage(d[lower:upper], count)
            r2, r2V, x, y = zipfs(wCount)
            sLen = avgSentenceLength(d[lower:upper])
            punc = pCount(d[lower:upper])
            cCount = contractionCount(d[lower:upper])
            avgWordLength = sum((len(x[0] * x[1]) for x in wCount)) / wordLim
            if yuleK > 0 and sLen > 0:
                # values.append([yuleK,yuleI,r2V,r2,punc,sLen,avgWordLength,cCount])
                values.append([yuleK, yuleI, r2V, r2, qPen,
                               punc, sLen, avgWordLength, cCount])
            if end:
                break
        f.close()
        #dFile.append(file)
        if values:
            ret.append(values)
        for val in values:
            dFile.append(val)
        values = []

        # df = pd.DataFrame(values, columns=[
        #                "YulesK", "YulesI", "Zipfs1", "Zipfs2", "Punc", "Sentence", "WordLen", "Contraction Usage"])
        # df = pd.DataFrame(dFile, columns=["YulesK", "YulesI", "Zipfs1", "Zipfs2",
        #                                "Quote", "Punc", "Sentence", "WordLen", "Contraction Usage"])
        # df.to_csv(folder + str(file)+".csv", index=False, header=False)
    #print(dFile)
    return ret


if __name__ == "__main__":
    a = 1
