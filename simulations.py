import numpy as np
import pandas as pd
import random
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)


class Node:
    def __init__(self):
        self.status = 0 # 0 = cooperate, 1 = defect
        self.wealth = 0
        self.lastPayoff = []

    def updatePayoff(self, newPayoff):
        self.wealth = self.wealth + newPayoff
        self.lastPayoff.append(newPayoff)

    def updateStatus(self, newStatus):
        self.status = newStatus

    def reset(self, numIterations):
        self.status = 0 # 0 = cooperate, 1 = defect
        self.wealth = 0
        self.lastPayoff = [0]* numIterations



class Simulation:
    def __init__(self, numIterations, saveRate, strategy):
        self.strategy = strategy
        self.numIterations = numIterations
        self.saveRate = saveRate
        self.savePath = os.path.join(dirName, 'savedInfo')
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def __call__(self, nodesDict, adjMatrix):
        if self.strategy == 0:
            nodesDict, adjMatrix = self.simulation_simplest(nodesDict, adjMatrix)
            return nodesDict, adjMatrix
        elif self.strategy == 1:
            return self.simulation_type1(nodesDict, adjMatrix)
        else:
            return self.simulation_type2(nodesDict, adjMatrix)

    def simulation_simplest(self, nodesDict, adjMatrix):
        numNodes = len(adjMatrix[0])
        for iter in range(self.numIterations):
            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]: # there is an edge
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j])
                        if nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0
                        else:
                            adjMatrix[i][j] = 1
                            adjMatrix[j][i] = 1
            self.saveOutput(nodesDict, adjMatrix, iter)

        return nodesDict, adjMatrix

    def simulation_type1(self, nodesDict, adjMatrix):
        numNodes = len(adjMatrix[0])
        for iter in range(self.numIterations):
            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]: # there is an edge
                        # Update edges
                        if nodesDict[i].status == 1 and nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0
                            otherNodes = list(range(numNodes))[:i] + list(range(numNodes))[i+1:]
                            newNodeToLink = random.choice(otherNodes)
                            adjMatrix[i][newNodeToLink] = 1
                            adjMatrix[newNodeToLink][i] = 1

                        elif nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0

                            if nodesDict[i].status == 1: # i defect
                                otherNodes = list(range(numNodes))[:j] + list(range(numNodes))[j+1:]
                                newNodeToLink = random.choice(otherNodes)
                                adjMatrix[j][newNodeToLink] = 1
                                adjMatrix[newNodeToLink][j] = 1 # j forms a new edge with another node
                            else:
                                otherNodes = list(range(numNodes))[:i] + list(range(numNodes))[i+1:]
                                newNodeToLink = random.choice(otherNodes)
                                adjMatrix[i][newNodeToLink] = 1
                                adjMatrix[newNodeToLink][i] = 1 # i forms a new edge with another node
                        else:
                            adjMatrix[i][j] = 1
                            adjMatrix[j][i] = 1

                        # Update node classes
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j])
                        nodesDict[i] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[i], iter)
                        nodesDict[j] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[j], iter)
            self.saveOutput(nodesDict, adjMatrix, iter)
        return nodesDict, adjMatrix


    def simulation_type2(self, nodesDict, adjMatrix):
        [node.reset(self.numIterations) for node in nodesDict.values()]
        numNodes = len(adjMatrix[0])
        for iter in range(self.numIterations):
            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]: # there is an edge
                        # Update edges
                        if nodesDict[i].status == 1 and nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0
                            otherNodes = list(range(numNodes))[:i] + list(range(numNodes))[i+1:]
                            newNodeToLink = random.choice(otherNodes)
                            adjMatrix[i][newNodeToLink] = 1
                            adjMatrix[newNodeToLink][i] = 1

                        elif nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            adjMatrix[i][j] = 0
                            adjMatrix[j][i] = 0

                            if nodesDict[i].status == 1: # i defect
                                otherNodes = list(range(numNodes))[:j] + list(range(numNodes))[j+1:]
                                newNodeToLink = random.choice(otherNodes)
                                adjMatrix[j][newNodeToLink] = 1
                                adjMatrix[newNodeToLink][j] = 1 # j forms a new edge with another node
                            else:
                                otherNodes = list(range(numNodes))[:i] + list(range(numNodes))[i+1:]
                                newNodeToLink = random.choice(otherNodes)
                                adjMatrix[i][newNodeToLink] = 1
                                adjMatrix[newNodeToLink][i] = 1 # i forms a new edge with another node
                        else:
                            adjMatrix[i][j] = 1
                            adjMatrix[j][i] = 1

                        # Update node classes
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j])
                        nodeiNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[i]) if isNeighbor]
                        nodejNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[j]) if isNeighbor]

                        nodesDict[i] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[i], nodeiNeighbors, iter)
                        nodesDict[j] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[j], nodejNeighbors, iter)
            self.saveOutput(nodesDict, adjMatrix, iter)
        return nodesDict, adjMatrix

    def updateNodesStatusBasedOnAvgPayoff(self, node, neighborsList, iter):
        neighborsLastPayoff = [neighbor.lastPayoff[iter] for neighbor in neighborsList]
        if node.lastPayoff[iter] < np.mean(neighborsLastPayoff):
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)
        return node

    def updateNodesStatusBasedOnLastPayoff(self, node, iter):
        if node.lastPayoff[iter] < node.lastPayoff[iter - 1]:
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)
        return node

    def gamePayoff(self, status1, status2):
        if status1 == 0 and status2 == 0: # both cooperate
            return 2, 2
        elif status1 == 0 and status2 == 1:
            return 0, 3
        elif status1 == 1 and status2 == 0:
            return 3, 0
        else:
            return 1, 1

    def updateWithGamePayoff(self, node1, node2):
        newPayoff1, newPayoff2 = self.gamePayoff(node1.status, node2.status)
        node1.updatePayoff(newPayoff1)
        node2.updatePayoff(newPayoff2)
        return node1, node2

    def saveOutput(self, nodesList, adjMatrix, iter):
        if iter % self.saveRate != 0 and iter!= self.numIterations - 1:
            return

        # save Nodes Info
        nodesInfo = [{'NodeID': nodeID, 'Status': nodesList[nodeID].status, 'Wealth': nodesList[nodeID].wealth,
                      'LastPayoff': nodesList[nodeID].lastPayoff} for nodeID in range(len(nodesList))]
        field_names = ['NodeID', 'Status', 'Wealth', 'LastPayoff']

        nodeFileName = 'nodeClassInfo_strategy{}_round{}.csv'.format(str(self.strategy), str(iter))
        with open(os.path.join(self.savePath, nodeFileName), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(nodesInfo)

        # save Adj mat Info
        adjFileName = 'adjMatrix_strategy{}_round{}.csv'.format(str(self.strategy), str(iter))
        pd.DataFrame(adjMatrix).to_csv(os.path.join(self.savePath, adjFileName))

        return
