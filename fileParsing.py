import textract
import re
import os
import sys


class Documents:

    def __init__(self, documents):
        self.documents = documents


def files():
    from tkinter import filedialog
    from tkinter import Tk

    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    files = os.listdir(folder)
    documents = []
    for file in files:
        if ".pdf" in file or ".docx" in file or ".doc" in file or ".odt" in file:
            documents.append(file)
    return documents, folder


def opening(path):
    isEmpty = False
    try:
        file = textract.process(path, encoding='utf-8')
        check = re.search(rb'[\s\n\r]\s*', file)
        check = file[:check.start()] + file[check.end():]
        if check.decode() == '':
            isEmpty = True
            sys.exit(1)
    except:
        if isEmpty:
            print("Document", path,
                  "is either blank or all images, contact author for new copy")
        else:
            print("Document", path, "Cannot be Read, Save it as a .docx")
        sys.exit(1)
    file = re.split(rb'[\s\n\r]\s*', file)
    count = 0
    for elem in file:
        # table of contents returns an error
        if (b"references" in elem.lower() or b"biliography" in elem.lower()) and count > 500:
            break
        else:
            count += 1
    if count != 0:
        file = file[:count]
    return file


def save(fname, newFile):
    f = open(fname + ".txt", "w")
    for word in newFile:
        try:
            f.write(str(word) + " ")
        except:
            pass
    f.close()


def createTxt():
    file, folder = files()
    count = 0
    for doc in file:

        if ".pdf" in doc or ".doc" in doc or ".docx" in doc or ".odt" in doc:
            f = opening(folder + "/" + doc)
            # print(f)
            if f:
                newFile = decoding(f)
                save(folder + "/" + doc[:doc.find(".")], newFile)
                count += 1
        else:
            continue
    return folder


def decoding(file):
    newFile = []
    a = []
    isQuote = False
    for i in range(0, len(file)):
        if b'\xe2\x80\x9c' in file[i] or isQuote:
            if b'\xe2\x80\x9c' in file[i] and isQuote:
                qCount += 1
                qSize += 1
                continue
            if not isQuote:
                qSize = 1
                qCount = 0
                isQuote = True
            if not b'\xe2\x80\x9d' in file[i]:
                qSize += 1
            else:

                if qCount == 0:
                    isQuote = False
                    if b"." in file[i]:
                        newFile.append("Quote(" + str(qSize) + ").")
                    else:
                        newFile.append("Quote(" + str(qSize) + ")")
                else:
                    qCount -= 1

        else:
            if not file[i].decode().replace('.', '').replace('-', '').isnumeric():
                newFile.append(file[i].decode())
    return newFile


def lenCheck(folder, documents):
    minLength = None
    for doc in documents:
        f = open(folder + "/" + doc, "r")
        data = f.read().split()
        if minLength and len(data) < minLength:
            minLength = len(data)
        elif not minLength:
            minLength = len(data)
    num = 0.124214 * (minLength) + 181.356
    num = round(num / 50) * 50
    return int(num)
