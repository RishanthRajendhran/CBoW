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
    "-trainDir",
    help="Path to directory containing train files",
    default="./data/train"
)

parser.add_argument(
    "-valDir",
    help="Path to directory containing validation files",
    default="./data/dev"
)

parser.add_argument(
    "-vocabFile",
    help="Path to file containing vocabulary",
    default="vocab.txt"
)

parser.add_argument(
    "-out",
    help="Path to directory where learned embedding should be saved",
    default="./embeddings/"
)

parser.add_argument(
    "-windowSize",
    type=int,
    help="Size of context window",
    default=5
)

parser.add_argument(
    "-embedDim",
    type=int,
    help="Size of word embedding",
    default=100
)

parser.add_argument(
    "-seed",
    type=int,
    help="Seed for torch/numpy",
    default=13
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=10
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=64
)

parser.add_argument(
    "-learningRate",
    type=float,
    nargs="+",
    help="Learning rate(s) for optimizer",
    default=[0.01, 0.01, 0.0001]
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=-1
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
class CBoW(torch.nn.Module):
    def __init__(self, windowSize, embedDim, vocab, device="cpu"):
        super(CBoW, self).__init__()
        self.windowSize = windowSize
        self.embedDim = embedDim
        self.vocab = vocab
        self.device = device
        self.projection = torch.nn.Embedding(
            num_embeddings=len(vocab), 
            embedding_dim=embedDim,
            max_norm=1,
        )
        self.classifier = torch.nn.Linear(
            in_features=embedDim, 
            out_features=len(vocab), 
        )
        self.to(device)

    def forward(self, x):
        x = x.to(device=self.device)
        embeddings = self.projection(x)
        embeddings = torch.sum(embeddings, axis=-2)
        out = self.classifier(embeddings)
        return out

    def to(self, device):
        self.device = device 
        self = super(CBoW, self).to(device)
        return self
    
    def getEmbeddings(self):
        return  self.classifier.weight.T.cpu().tolist()

    def getEmbedding(self, word):
        with torch.no_grad():
            vec = torch.zeros((len(self.vocab),))
            vec[word] = 1
            vec = vec.to(self.device)
            return  np.dot(vec.cpu(), self.classifier.weight.cpu()).tolist()
#---------------------------------------------------------------------------
class CBoWDataset:
    def __init__(self, data, windowSize, vocabSize):
        self.data = data
        self.vocabSize = vocabSize

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        curInstance = {}
        curInstance["context"] = self.data[item]["context"]
        #Not needed when using torch.nn.Embedding
        # curInstance["context"] = [self.oneHot(ind, self.vocabSize) for ind in curInstance["context"]]
        curInstance["context"] = torch.tensor(curInstance["context"])
        curInstance["label"] = torch.tensor(self.data[item]["label"])
        return curInstance["context"], curInstance["label"]
    
    def oneHot(self, ind, size):
        vec = torch.zeros((size,))
        vec[ind] = 1
        return vec.tolist()
#---------------------------------------------------------------------------
def collateBatch(batch):
    contexts, labels = zip(*batch)
    contexts = torch.stack(contexts)
    labels = torch.stack(labels)
    return contexts, labels
#---------------------------------------------------------------------------
def createDataLoader(data, windowSize, batchSize, vocabSize):
    ds = CBoWDataset(
        data = data, 
        windowSize=windowSize,
        vocabSize=vocabSize,
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        collate_fn=collateBatch,
    )
#---------------------------------------------------------------------------
def get_word2ix(path = "./../vocab.txt"):
    word2ix = {}
    with open(path) as f:
        data = f.readlines()
        for line in data:
            word2ix[line.split("\t")[1].strip()] = int(line.split("\t")[0])
    
    return word2ix
#---------------------------------------------------------------------------
def get_files(path):
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list
#---------------------------------------------------------------------------
def process_data(files, context_window, word2ix):
    data = []
    for file in files:
        file_data = [word2ix["[PAD]"]]*context_window
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() not in word2ix.keys():
                    file_data.append(word2ix["[UNK]"])
                else:
                    file_data.append(word2ix[line.strip()])
            
            file_data.extend([word2ix["[PAD]"]]*context_window)
            data.append(file_data.copy())
    return data
#---------------------------------------------------------------------------
def trainModel(model, dataLoader, lossFunction, optimizer, device, scheduler=None, maxSteps=-1, logSteps=1000):
    model.to(device)
    model.train()

    losses = []
    corrPreds = 0
    numExamples = 0
    numBatch = 0
    numSteps = 0
    for contexts, labels in tqdm(dataLoader, desc="Train data"):
        numBatch += 1
        numExamples += len(contexts)
        outputs = model(contexts)
        labels = labels.to(device)
        
        _, preds = torch.max(outputs, dim=-1)

        loss = lossFunction(outputs, labels)

        if numSteps%logSteps == 0:
            logging.info(f"\nBatch: {numBatch}/{len(dataLoader)}, Loss: {loss.item()}")

        corrPreds += torch.sum(preds == labels)
        losses.append(loss.item())
        #Zero out gradients from previous batches
        optimizer.zero_grad()
        #Backwardpropagate the losses
        loss.backward()
        # #Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #Perform a step of optimization
        optimizer.step()
        numSteps += 1
        if maxSteps and numSteps >= maxSteps:
            break
    if scheduler:
        scheduler.step()
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def evalModel(model, lossFunction, dataLoader, device="cpu", dataDesc="Test batch"):
    model.eval()
    with torch.no_grad():
        losses = []
        corrPreds = 0
        numExamples = 0
        numBatch = 0
        numSteps = 0
        for contexts, labels in tqdm(dataLoader, desc="Train data"):
            numBatch += 1
            numExamples += len(contexts)
            outputs = model(contexts)
            labels = labels.to(device)
            _, preds = torch.max(outputs, dim=-1)

            loss = lossFunction(outputs, labels)

            corrPreds += torch.sum(preds == labels)
            losses.append(loss.item())
            numSteps += 1
    return corrPreds.double()/numExamples, np.mean(losses)
#---------------------------------------------------------------------------
def extractData(data, windowSize):
    #Assuming padding has been already done at the beggining and end
    extractedData = []
    for i in range(len(data)):
        for j in range(windowSize, len(data[i])-windowSize):
            curInstance  = {}
            curInstance["context"] = []
            curInstance["context"].extend(data[i][j-windowSize:j])
            curInstance["context"].extend(data[i][j+1:j+1+windowSize])
            curInstance["label"] = data[i][j]
            extractedData.append(curInstance)
    return extractedData
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

    if args.windowSize <= 0:
        raise ValueError("[main] Window size has to be positive!")
    
    saveModelPath = "./models/"
    
    args.out = checkIfExists(args.out, isDir=True, createIfNotExists=True)
    args.trainDir = checkIfExists(args.trainDir, isDir=True, createIfNotExists=False)
    args.valDir = checkIfExists(args.valDir, isDir=True, createIfNotExists=False)
    _ = checkIfExists(saveModelPath, isDir=True, createIfNotExists=True)
    _ = checkIfExists(args.vocabFile, isDir=False, createIfNotExists=False)
    checkFile(args.vocabFile, fileExtension=".txt")

    word2ix = get_word2ix(path=args.vocabFile)
    trainFiles = get_files(args.trainDir)
    valFiles = get_files(args.valDir)

    trainData = process_data(trainFiles, args.windowSize, word2ix)
    valData = process_data(valFiles, args.windowSize, word2ix)

    trainData = extractData(trainData, args.windowSize)
    valData = extractData(valData, args.windowSize)

    if torch.cuda.is_available:
        device = "cuda"
    else: 
        device = "cpu"
    logging.info("Using device:{}".format(device))

    model = CBoW(args.windowSize, args.embedDim, word2ix, device=device)

    trainDataLoader = createDataLoader(trainData, args.windowSize, args.batchSize, len(word2ix))
    valDataLoader = createDataLoader(valData, args.windowSize, args.batchSize, len(word2ix))

    numTrainingSteps = args.numEpochs * len(trainDataLoader)
    maxSteps = args.maxSteps
    if maxSteps == -1:
        maxSteps = numTrainingSteps
    elif maxSteps > 0:
        maxSteps = math.ceil(maxSteps/len(trainDataLoader))
    else: 
        raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")

    bestLearningRate = None
    bestValLoss = None
    bestValAcc = 0
    for learningRate in args.learningRate:
        logging.info("Learning Rate: {}".format(learningRate))
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learningRate, 
            weight_decay=args.weightDecay,
        )
        totalSteps = args.numEpochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            # num_warmup_steps=0.075*totalSteps,
            # num_warmup_steps=2000,
            num_warmup_steps=0,
            num_training_steps=totalSteps
        )
        # scheduler = None
        lossFunction = torch.nn.CrossEntropyLoss().to(device)
        
        for epoch in range(args.numEpochs):
            curAcc, curLoss = trainModel(model, trainDataLoader, lossFunction, optimizer, device, scheduler, maxSteps)
            maxSteps -= len(trainDataLoader)
            valAcc, valLoss = evalModel(
                model, 
                lossFunction,
                valDataLoader, 
                device=device,
                dataDesc="Validation batch", 
            )

            logging.info("Epoch {}/{}\nTraining Loss: {:0.2f}\nTrain Accuracy: {:0.2f}%\nValidation Loss: {:0.2f}\nValidation Accuracy: {:0.2f}%".format(epoch+1, args.numEpochs, curLoss, curAcc*100, valLoss, valAcc*100))
            logging.info("*****")

            if not bestValLoss or bestValLoss >= valLoss:
                bestValLoss = valLoss
                bestValAcc = valAcc
                bestLearningRate = learningRate
                torch.save(model, f"{saveModelPath}model.pt")
                logging.info("Model saved at '{}model.pt'".format(saveModelPath))
            if maxSteps <= 0:
                break
    logging.info("Best learning rate: {}".format(bestLearningRate))
    logging.info("Best model's validation loss: {}".format(bestValLoss))
    logging.info("Best model's validation accuracy: {:0.2f}%".format(bestValAcc*100))

    model = torch.load(f"{saveModelPath}model.pt")
    with open(f"{args.out}embeddings.txt","w") as f:
        f.write("{} {}".format(len(word2ix), args.embedDim))
        f.write("\n")
        for word, index  in word2ix.items():
            embedding = model.getEmbedding(index)
            f.write(word)
            f.write(" ")
            f.write(" ".join([str(e) for e in embedding]))
            f.write("\n")
#------------------- --------------------------------------------------------
if __name__=="__main__":
    main()