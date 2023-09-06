import argparse
import logging
import numpy as np
from pathlib import Path
from os.path import exists
import os
import glob
import torch
import pandas as pd
import transformers
import math
from tqdm import tqdm
import argparse
from gensim.models import KeyedVectors
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt
import itertools

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-out",
    help="Path to directory where evaluation outputs are to be saved",
    default="./outputs/"
)

parser.add_argument(
    "-seed",
    type=int,
    help="Seed for torch/numpy",
    default=13
)

parser.add_argument(
    "-eFile",
    "--emb_file",
    help="Absolute path to embeddings file",
    required=True
)

parser.add_argument(
    "-wordSim",
    action="store_true",
    help="Boolean flag to enable word similarity evaluation"
)

parser.add_argument(
    "-wsFile",
    help="Path to file containing pairs for which word-similarity computations/comparisions are to be performed; In case of computation: one-pair pair line of the form 'word1:word2'; In case of comparisons: two-pairs separated by a comma of the form 'word11:word12,word21:word22'"
)

parser.add_argument(
    "-analogy",
    action="store_true",
    help="Boolean flag to enable analogy evaluation"
)

parser.add_argument(
    "-aFile",
    help="Path to file containing pairs for which analogy tests are to be performed; Expected format: 'word11:word12,word21:?'"
)

parser.add_argument(
    "-wordsToVis",
    help="Path to file containing words (one word per line) to visualize"
)
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        path += "/"
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")   
    return path
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#---------------------------------------------------------------------------
def readFile(fileName):
    data = []
    if fileName.endswith(".txt"):
        with open(fileName, "r") as f: 
            data = list(f.readlines()) 
    else: 
        raise ValueError("[readFile] Unsupported file extension: .{}".format(fileName.split(".")[-1]))
    return data
#---------------------------------------------------------------------------
def writeFile(data, fileName):
    if fileName.endswith(".txt"):
        with open(fileName, "w") as f: 
            for d in data: 
                f.write(d)
                f.write("\n")
    else: 
        raise ValueError("[writeFile] Unsupported file extension: .{}".format(fileName.split(".")[-1]))
#---------------------------------------------------------------------------
def getEmbeddings(data):
    N, E = data[0].split(" ")
    N, E = int(N), int(E) 
    data = data[1:]
    wordToInd, matrix = {}, []
    for line in data: 
        splitLine = line.split(" ")
        wordToInd[splitLine[0]] = len(matrix)
        matrix.append([float(e) for e in splitLine[1:]])
    if len(wordToInd)!=N:
        raise RuntimeError("[getEmbeddings] Could not find {} (!= {}) words in the input embeddings file!".format(N, len(wordToInd)))
    if len(matrix[0])!=E:
        raise RuntimeError("[getEmbeddings] Vectors are not of size {} (!= {}) in the input embeddings file!".format(E, len(matrix[0])))
    return wordToInd, matrix
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    args.out = checkIfExists(args.out, isDir=True, createIfNotExists=True)
    checkFile(args.emb_file, ".txt")

    if args.wordSim:
        wv = KeyedVectors.load_word2vec_format(args.emb_file, binary=False)
        result = []
        if args.wsFile:
            checkFile(args.wsFile, ".txt")
            data = readFile(args.wsFile) 
            for i, d in enumerate(data):
                if len(d) == 0:
                    continue
                words = d.split(",")
                if len(words) == 1:
                    words = words[0].split(":")
                    if len(words) == 2:
                        words = [w.strip() for w in words]
                        try:
                            sim = wv.similarity(*words)
                        except: 
                            result.append("Could not find one or more words in vocabulary: {}".format(*words))
                        result.append("similarity({}, {}) = {}".format(*words, sim))
                    else:
                        try:
                            simWord, sim = wv.most_similar(positive=[words[0]])[0]
                            result.append("similarity({}, {}) = {}".format(words[0], simWord, sim))
                        except: 
                            result.append("{} not in vocab".format(words[0]))
                elif len(words) == 2:
                    words = [w.split(":") for w in words]
                    if len(words[0]) != 2:
                        simWord, sim1 = wv.most_similar(positive=[words[0][0]])[0]
                        words[0] = [words[0][0], simWord]
                    else:
                        words[0] = [w.strip() for w in words[0]]
                        try:
                            sim1 = wv.similarity(*words[0]) 
                        except: 
                            result.append("Could not find one or more words in vocabulary: {}".format(*words[0]))
                    if len(words[1]) != 2:
                        simWord, sim2 = wv.most_similar(positive=[words[1][0]])[0]
                        words[1] = [words[1][0], simWord]
                    else:
                        words[1] = [w.strip() for w in words[1]]
                        try:
                            sim2 = wv.similarity(*words[1]) 
                        except: 
                            result.append("Could not find one or more words in vocabulary: {}".format(*words[1]))
                    if sim1 >= sim2:
                        result.append("({}, {}) [similarity({}, {}) = {}, similarity({}, {}) = {}]".format(*words[0], *words[0], sim1, *words[1], sim2)) 
                    else: 
                        result.append("({}, {}) [similarity({}, {}) = {}, similarity({}, {}) = {}]".format(*words[1], *words[0], sim1, *words[1], sim2)) 
                else: 
                    raise RuntimeError("[main] Found more than one comma (,) in line {} of wsFile!".format(i+1))
            writeFile(result, "{}wordSimilarity_{}.txt".format(args.out, args.wsFile.split("/")[-1].split(".")[0]))
        else: 
            checkFile(args.emb_file, ".txt")
            embeddings = readFile(args.emb_file)
            wordToInd, matrix =  getEmbeddings(embeddings) 
            wordsToChooseFrom = list(wordToInd.keys())
            chosenWords1 = np.random.choice(wordsToChooseFrom, 50, replace=False)
            chosenWords2 = np.random.choice(np.setdiff1d(wordsToChooseFrom, chosenWords1), 50, replace=False)
            for word1, word2 in zip(chosenWords1, chosenWords2):
                simWord1, sim1 = wv.most_similar(positive=[word1])[0]
                simWord2, sim2 = wv.most_similar(positive=[word2])[0]
                if sim1 >= sim2:
                    result.append("({}, {}) [similarity({}, {}) = {}, similarity({}, {}) = {}]".format(word1, simWord1, word1, simWord1, sim1, word2, simWord2, sim2)) 
                else: 
                    result.append("({}, {}) [similarity({}, {}) = {}, similarity({}, {}) = {}]".format(word2, simWord2, word1, simWord1, sim1, word2, simWord2, sim2)) 
            writeFile(result, f"{args.out}wordSimilarityRandom.txt")
    elif args.analogy:
        checkFile(args.emb_file, ".txt")
        wv = KeyedVectors.load_word2vec_format(args.emb_file, binary=False)
        embeddings = readFile(args.emb_file)
        wordToInd, matrix =  getEmbeddings(embeddings) 
        result = []
        if args.aFile:
            checkFile(args.aFile, ".txt")
            data = readFile(args.aFile) 
            for i, d in enumerate(data):
                if len(d) == 0:
                    continue
                words = d.split(",")
                if len(words) == 2:
                    words = [w.split(":") for w in words]
                    if len(words[0]) != 2:
                        raise RuntimeError("[main] Could not find colon (:) in line {} in aFile!".format(i+1))
                    word1, word2, word3 = words[0][0], words[0][1], words[1][0]
                    if word1 not in wordToInd.keys():
                        result.append("{} not in vocab".format(word1))
                        continue
                    if word2 not in wordToInd.keys():
                        result.append("{} not in vocab".format(word2))
                        continue
                    if word3 not in wordToInd.keys():
                        result.append("{} not in vocab".format(word3))
                        continue
                    vec1, vec2, vec3 = matrix[wordToInd[word1]], matrix[wordToInd[word2]], matrix[wordToInd[word3]]
                    vec1 = np.array(vec1)
                    vec2 = np.array(vec2)
                    vec3 = np.array(vec3)
                    ind = 0
                    simWord, sim = wv.similar_by_vector(vec3+(vec1-vec2))[ind]
                    while simWord in [word1, word2, word3]:
                        ind += 1
                        simWord, sim = wv.similar_by_vector(vec3+(vec1-vec2))[ind]
                    result.append("{}:{},{}:{} [{}]".format(word1, word2, word3, simWord, sim))
                else: 
                    raise RuntimeError("[main] Could not find a comma (,) in line {} in aFile!".format(i+1))
            writeFile(result, "{}analogyPair_{}.txt".format(args.out, args.aFile.split("/")[-1].split(".")[0]))
        else: 
            checkFile(args.emb_file, ".txt")
            embeddings = readFile(args.emb_file)
            wordToInd, matrix =  getEmbeddings(embeddings) 
            wordsToChooseFrom = list(wordToInd.keys())
            chosenWords1 = np.random.choice(wordsToChooseFrom, 50, replace=False)
            chosenWords2 = np.random.choice(np.setdiff1d(wordsToChooseFrom, chosenWords1), 50, replace=False)
            for word1, word3 in zip(chosenWords1, chosenWords2):
                word2, _ = wv.most_similar(positive=[word1])[0]
                vec1, vec2, vec3 = matrix[wordToInd[word1]], matrix[wordToInd[word2]], matrix[wordToInd[word3]]
                vec1 = np.array(vec1)
                vec2 = np.array(vec2)
                vec3 = np.array(vec3)
                ind = 0
                simWord, sim = wv.similar_by_vector(vec3+(vec1-vec2))[ind]
                while simWord in [word1, word2, word3]:
                    ind += 1
                    simWord, sim = wv.similar_by_vector(vec3+(vec1-vec2))[ind]
                result.append("{}:{},{}:{} [{}]".format(word1, word2, word3, simWord, sim))
            writeFile(result, f"{args.out}analogyPairRandom.txt")
    else: 
        checkFile(args.emb_file, ".txt")
        embeddings = readFile(args.emb_file)
        wordToInd, matrix =  getEmbeddings(embeddings)
        svd = TruncatedSVD(n_components=2, n_iter=20, random_state=args.seed)
        # svd = PCA(n_components=2, random_state=args.seed)
        newMatrix =  svd.fit_transform(matrix)
        checkFile(args.wordsToVis, ".txt")
        wordsToVis = readFile(args.wordsToVis)
        vis, labels = [], []
        for word in wordsToVis:
            word = word.strip()
            if word in wordToInd.keys():
                vis.append(newMatrix[wordToInd[word]])
                labels.append(word)
            else: 
                word = word.lower()
                if word in wordToInd.keys():
                    vis.append(newMatrix[wordToInd[word]])
                    labels.append(word)
                else:
                    logging.warning("[main] Could not find word in vocabulary: {}".format(word))
        vis = np.array(vis)
        plt.scatter(vis[:,0], vis[:,1])
        for i, word in enumerate(labels):
            plt.annotate(word, (vis[i][0], vis[i][1]+0.05))
        plt.savefig(f"{args.out}2dProjections.png")
#----------------------------------------------------------------------------
if __name__=="__main__":
    main()