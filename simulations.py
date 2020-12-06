import numpy as np
import pandas as pd
import random
import csv
import os
import pickle
from copy import deepcopy
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)

class CalcPayoff:
    def __init__(self, payoffID):
        self.payoffID = payoffID

    def __call__(self, status1, status2):
        if self.payoffID == 0:
            if status1 == 0 and status2 == 0:  # both cooperate
                return 2, 2
            elif status1 == 0 and status2 == 1:
                return -1, 1
            elif status1 == 1 and status2 == 0:
                return 1, -1
            else:
                return -1, -1

        elif self.payoffID == 1:
            if status1 == 0 and status2 == 0:  # both cooperate
                return 1, 1
            elif status1 == 0 and status2 == 1:
                return -1, 1
            elif status1 == 1 and status2 == 0:
                return 1, -1
            else:
                return -1, -1

        else:
            if status1 == 0 and status2 == 0:  # both cooperate
                return 0, 0
            elif status1 == 0 and status2 == 1:
                return -1, 1
            elif status1 == 1 and status2 == 0:
                return 1, -1
            else:
                return -1, -1

class Node:
    def __init__(self, status):
        self.status = status # 0 = cooperate, 1 = defect
        self.wealth = 0
        self.lastPayoff = []

    def updatePayoff(self, newPayoff):
        self.wealth = self.wealth + newPayoff
        self.lastPayoff.append(newPayoff)

    def updateStatus(self, newStatus):
        self.status = newStatus


class Simulation:
    def __init__(self, numIterations, saveRate, strategy, payoffID):
        self.strategy = strategy
        self.numIterations = numIterations
        self.payoffID = payoffID
        self.calcPayoff = CalcPayoff(payoffID)
        self.saveRate = saveRate
        self.savePath = os.path.join(dirName, 'savedInfo')
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def __call__(self, nodesDict, adjMatrix):
        if self.strategy == 0:
            return self.simulation_simplest(nodesDict, adjMatrix)
        else:
            return self.simulation_unweighted(nodesDict, adjMatrix, self.strategy)

    def simulation_simplest(self, nodesDict, adjMatrix, pInteract = .1):
        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)

        numNodes = len(adjMatrix[0])
        for iter in range(self.numIterations):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j] and random.uniform(0, 1) < pInteract: # there is an edge then interact with prob = p
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j])
                        if nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0
                            print('changed')
                        else:
                            adjMatrix[i][j] = 1
                            adjMatrix[j][i] = 1

            nodesDict_list.append(nodesDict)
            adjMatrix_list.append(adjMatrix)

        return nodesDict_list, adjMatrix_list

    def linkToNewNode(self, adjMatrix, nodeID, excludingNodesID):
        otherNodesID = [otherNodeID for otherNodeID, isNeighbor in enumerate(adjMatrix[nodeID])
                        if not isNeighbor and otherNodeID not in excludingNodesID]
        if len(otherNodesID) == 0:
            return adjMatrix

        newNodeToLink = random.choice(otherNodesID)
        adjMatrix[nodeID][newNodeToLink] = 1
        adjMatrix[newNodeToLink][nodeID] = 1
        return adjMatrix

    def simulation_unweighted(self, nodesDict, adjMatrix, type):
        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)  # adding initial structure
        numNodes = len(adjMatrix[0])

        for iterID in tqdm(range(self.numIterations)):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]:  # there is an edge
                        # Update edges
                        if nodesDict[i].status == 1 and nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0
                            excludingNodesID = [i, j]
                            adjMatrix = self.linkToNewNode(adjMatrix, i, excludingNodesID)

                        elif nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0
                            excludingNodesID = [i, j]

                            adjMatrix = self.linkToNewNode(adjMatrix, j, excludingNodesID) if nodesDict[i].status == 1 \
                                else self.linkToNewNode(adjMatrix, i, excludingNodesID)

                        else:
                            pass

                        # Update node classes
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j])

                        if type == 1:
                            nodesDict[i] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[i])
                            nodesDict[j] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[j])

                        elif type == 2:
                            nodeiNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[i]) if isNeighbor]
                            nodejNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[j]) if isNeighbor]
                            nodesDict[i] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[i], nodeiNeighbors)
                            nodesDict[j] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[j], nodejNeighbors)

            nodesDict_list.append(nodesDict)
            adjMatrix_list.append(adjMatrix)

        self.saveOutput(nodesDict_list, adjMatrix_list)
        return nodesDict_list, adjMatrix_list


    def simulation_unweighted_withMultiEdge(self, nodesDict, adjMatrix, type):
        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)    # adding initial structure
        numNodes = len(adjMatrix[0])

        for iterID in range(self.numIterations):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]: # there is an edge
                        # Update edges
                        if nodesDict[i].status == 1 and nodesDict[j].status == 1:
                            adjMatrix[i][j] -= 1
                            adjMatrix[j][i] -= 1
                            otherNodes = list(range(numNodes))[:i] + list(range(numNodes))[i+1:]
                            newNodeToLink = random.choice(otherNodes)
                            adjMatrix[i][newNodeToLink] += 1
                            adjMatrix[newNodeToLink][i] += 1

                        elif nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            adjMatrix[i][j] -= 1
                            adjMatrix[j][i] -= 1

                            if nodesDict[i].status == 1: # i defect
                                otherNodes = list(range(numNodes))[:j] + list(range(numNodes))[j+1:]
                                newNodeToLink = random.choice(otherNodes)
                                adjMatrix[j][newNodeToLink] += 1
                                adjMatrix[newNodeToLink][j] += 1 # j forms a new edge with another node
                            else:
                                otherNodes = list(range(numNodes))[:i] + list(range(numNodes))[i+1:]
                                newNodeToLink = random.choice(otherNodes)
                                adjMatrix[i][newNodeToLink] += 1
                                adjMatrix[newNodeToLink][i] += 1 # i forms a new edge with another node
                        else:
                            # adjMatrix[i][j] = 1
                            # adjMatrix[j][i] = 1
                            pass

                        # Update node classes
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j])
                        
                        if type == 1:
                            nodesDict[i] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[i])
                            nodesDict[j] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[j])
                        
                        elif type == 2:
                            nodeiNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[i]) if isNeighbor]
                            nodejNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[j]) if isNeighbor]
                            nodesDict[i] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[i], nodeiNeighbors)
                            nodesDict[j] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[j], nodejNeighbors)

            if iterID % self.saveRate == 0 or iterID == (self.numIterations - 1):
                nodesDict_list.append(nodesDict)
                adjMatrix_list.append(adjMatrix)

            self.saveOutput(nodesDict_list, adjMatrix_list)

        return nodesDict_list, adjMatrix_list

    def updateNodesStatusBasedOnAvgPayoff(self, node, neighborsList):
        neighborsLastPayoff = [neighbor.lastPayoff[-1] for neighbor in neighborsList if (len(neighbor.lastPayoff) > 0)]
        mean = np.mean(neighborsLastPayoff) if len(neighborsLastPayoff) > 0 else 0
        if node.lastPayoff[-1] < mean:
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)
        return node

    def updateNodesStatusBasedOnLastPayoff(self, node):
        if len(node.lastPayoff) <= 2:
            return node

        if node.lastPayoff[-1] < node.lastPayoff[-2]:
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)
        return node

    def updateWithGamePayoff(self, node1, node2):
        newPayoff1, newPayoff2 = self.calcPayoff(node1.status, node2.status)
        node1.updatePayoff(newPayoff1)
        node2.updatePayoff(newPayoff2)
        return node1, node2

    def saveIndividualOutputToCsv(self, nodesDictList, adjMatrixList):
        numNodes = len(adjMatrixList[0][0])
        saveRoundTot = len(adjMatrixList)
        for saveRound, nodeDict, adjMatrix in zip(range(saveRoundTot), nodesDictList, adjMatrixList):
            iterID = saveRound* self.saveRate
            nodesInfo = [{'NodeID': nodeID, 'Status': nodeDict[nodeID].status, 'Wealth': nodeDict[nodeID].wealth,
                          'LastPayoff': nodeDict[nodeID].lastPayoff} for nodeID in range(numNodes)]
            field_names = ['NodeID', 'Status', 'Wealth', 'LastPayoff']
            nodeFileName = 'nodeClassInfo_strategy{}_round{}.csv'.format(str(self.strategy), str(iterID))

            with open(os.path.join(self.savePath, nodeFileName), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(nodesInfo)

            # save Adj mat Info
            adjFileName = 'adjMatrix_strategy{}_round{}.csv'.format(str(self.strategy), str(iterID))
            pd.DataFrame(adjMatrix).to_csv(os.path.join(self.savePath, adjFileName))

    def saveOutput(self, nodesDict_list, adjMatrix_list):
        # save nodes dict
        path = os.path.join(self.savePath, 'nodesDict_strategy{}payoff{}saveRate{}.pickle'.format(self.strategy, self.payoffID, self.saveRate))
        pklFile = open(path, "wb")
        pickle.dump(nodesDict_list, pklFile)
        pklFile.close()

        # save adjmatrix
        path = os.path.join(self.savePath, 'adjMat_strategy{}payoff{}saveRate{}.pickle'.format(self.strategy, self.payoffID, self.saveRate))
        pklFile = open(path, "wb")
        pickle.dump(adjMatrix_list, pklFile)
        pklFile.close()