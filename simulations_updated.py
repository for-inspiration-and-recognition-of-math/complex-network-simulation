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
        if self.payoffID == 0:  # using original prisoner's dilemma matrix
            if status1 == 0 and status2 == 0:  # both cooperate
                return -1, -1
            elif status1 == 0 and status2 == 1:
                return -3, 0
            elif status1 == 1 and status2 == 0:
                return 0, -3
            else:
                return -2, -2

        elif self.payoffID == 1: # favor cooperation
            if status1 == 0 and status2 == 0:  # both cooperate
                return 0, 0
            elif status1 == 0 and status2 == 1:
                return -3, 0
            elif status1 == 1 and status2 == 0:
                return 0, -3
            else:
                return -2, -2

        else: # favor defector
            if status1 == 0 and status2 == 0:  # both cooperate
                return -1.5, -1.5
            elif status1 == 0 and status2 == 1:
                return -3, 0
            elif status1 == 1 and status2 == 0:
                return 0, -3
            else:
                return -2, -2


class Node:
    def __init__(self, status, nodeID):
        self.status = status  # 0 = cooperate, 1 = defect
        self.lastPayoff = []
        self.statusHist = []
        self.nodeID = nodeID
        self.wealth = 0

    def updatePayoff(self, newPayoff, time_stamp):
        self.wealth = self.wealth + newPayoff
        self.lastPayoff[time_stamp].append(newPayoff)  # [[2, 0, 3], [1, 2], [], [3]]

    def updateStatus(self, newStatus):
        self.status = newStatus

    def updateStatusHistory(self, status):
        self.statusHist.append(status)

    def initalizeLastPayoffList(self, numIterations):
        self.lastPayoff = [[] for i in range(numIterations)]

    def getDefectPercent(self):
        return np.mean(self.statusHist)


class Simulation:
    def __init__(self, numIterations, saveRate, strategy, payoffID, fileName):
        self.strategy = strategy
        self.numIterations = numIterations
        self.payoffID = payoffID
        self.calcPayoff = CalcPayoff(payoffID)
        self.fileName = fileName
        self.saveRate = saveRate
        self.savePath = os.path.join(dirName, 'savedInfo')
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def __call__(self, nodesDict, adjMatrix):
        for node in nodesDict.values():
            node.initalizeLastPayoffList(self.numIterations)

        if self.strategy == 0:
            return self.simulation_simplest(nodesDict, adjMatrix)
        elif self.strategy == 1 or self.strategy == 2:
            return self.simulateEdgeByLastInteraction(nodesDict, adjMatrix, self.strategy)
        else:
            return self.simulateByStatusHistory(nodesDict, adjMatrix, self.strategy)


    def simulation_simplest(self, nodesDict, adjMatrix, pInteract=.1):
        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)
        numNodes = len(adjMatrix[0])

        for iterID in range(self.numIterations):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            brokenEdges = []

            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j] and random.uniform(0, 1) < pInteract:  # there is an edge then interact with prob = p
                        if nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            brokenEdges.append((i, j))

                        # Update node payoff history
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j], iterID)

            # break edges:
            for edge in brokenEdges:
                stub1, stub2 = edge
                adjMatrix[stub1][stub2] = 0
                adjMatrix[stub2][stub1] = 0

            nodesDict_list.append(nodesDict)
            adjMatrix_list.append(adjMatrix)

        self.saveOutput(nodesDict_list, adjMatrix_list)
        return nodesDict_list, adjMatrix_list

    def linkToNewNode(self, adjMatrix, excludingNodesIDDict, numNewEdgesToFormDict):
        for nodeID in excludingNodesIDDict.keys():
            otherNodesID = [otherNodeID for otherNodeID, isNeighbor in enumerate(adjMatrix[nodeID])
                            if (not isNeighbor)
                            and (otherNodeID not in excludingNodesIDDict[nodeID])
                            and (nodeID not in excludingNodesIDDict.get(otherNodeID, [-1]))]
            # form edge with you if
            # 1. you are not my neighbor
            # 2. I don't hate you : otherNodeID not in excludingNodesIDDict[nodeID]
            # 3. you don't hate me: (nodeID not in excludingNodesIDDict.get(otherNodeID, [-1]))
            # [-1]: taking care of you not hating anyone (empty dict)

            if len(otherNodesID) == 0:
                return adjMatrix

            numNewEdgesToForm = numNewEdgesToFormDict[nodeID]
            numNewEdgesToForm = numNewEdgesToForm if len(otherNodesID) >= numNewEdgesToForm else len(otherNodesID)
            newNodesToLink = np.random.choice(otherNodesID, numNewEdgesToForm, replace=False)
            for newNodeID in newNodesToLink:
                adjMatrix[nodeID][newNodeID] = 1
                adjMatrix[newNodeID][nodeID] = 1

        return adjMatrix


    # Node 0: 0.8 cooperation history -- 0.8 probability of being selected
    # Node 1: 0 cooperation history -- 0.1 probability of being selected

    # strategy 0: simplest
    # strategy 1: break edges = None, change status =
    # strategy 1: break edges based on whether defected last time + change status based on

    def simulateEdgeByLastInteraction(self, nodesDict, adjMatrix, strategy):
        numPunishRounds = 1 if strategy == 1 else 5

        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)  # adding initial structure
        numNodes = len(adjMatrix[0])
        nodesToAvoidInfo = {}  # {nodeWhoFormsNewEdge: {nodeToAvoid: remainingTimesToAvoid}}

        for iterID in tqdm(range(self.numIterations)):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            brokenEdges = []
            nodesToFornNewEdges = []

            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]:  # there is an edge
                        if nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            brokenEdges.append((i, j))

                            if nodesDict[j].status == 1:
                                nodesToFornNewEdges.append(i)
                                nodesToAvoidInfo = self.getNodesToAvoidWithMemory(nodesToAvoidInfo, i, j, numPunishRounds)
                            else:
                                nodesToFornNewEdges.append(j)
                                nodesToAvoidInfo = self.getNodesToAvoidWithMemory(nodesToAvoidInfo, j, i, numPunishRounds)

                        # Update node payoff history
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j], iterID)

            adjMatrixBeforeChanging = adjMatrix.copy()
            nodesDictOld = nodesDict.copy()  # create a copy for reference

            # break edges:
            for edge in brokenEdges:
                stub1, stub2 = edge
                adjMatrix[stub1][stub2] = 0
                adjMatrix[stub2][stub1] = 0

            # form new edges
            numNewEdgesToFormDict = {}
            excludingNodesIDDict = {}
            for nodeIDToCreateNewEdge in nodesToAvoidInfo:
                nodesToAvoidDict = nodesToAvoidInfo[nodeIDToCreateNewEdge]
                otherNodedToAvoid = [nodeID for nodeID in nodesToAvoidDict.keys() if nodesToAvoidDict[nodeID] != 0]  # punishment not over
                excludingNodesID = otherNodedToAvoid + [nodeIDToCreateNewEdge]  # no edges with nodes just interacted, no self-edge
                numNewEdges = len(otherNodedToAvoid)

                excludingNodesIDDict[nodeIDToCreateNewEdge] = excludingNodesID
                numNewEdgesToFormDict[nodeIDToCreateNewEdge] = numNewEdges

            adjMatrix = self.linkToNewNode(adjMatrix, excludingNodesIDDict, numNewEdgesToFormDict)

            # update nodes status
            for nodeID in nodesDict.keys():
                if strategy == 1:
                    nodesDict[nodeID] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[nodeID])

                elif strategy == 2:
                    nodeiNeighbors = [node for node, isNeighbor in zip(nodesDictOld.values(), adjMatrixBeforeChanging[nodeID]) if isNeighbor]
                    nodesDict[nodeID] = self.updateNodesStatusBasedOnNeighborsInfo(nodesDictOld[nodeID], nodeiNeighbors)

            # update nodes punishment count
            for nodeIDToCreateNewEdge in nodesToAvoidInfo.keys():
                for punishedNodeID in nodesToAvoidInfo[nodeIDToCreateNewEdge]:
                    currentPunishment = nodesToAvoidInfo[nodeIDToCreateNewEdge][punishedNodeID]
                    nodesToAvoidInfo[nodeIDToCreateNewEdge][punishedNodeID] = currentPunishment - 1 if currentPunishment > 0 else 0

            nodesDict_list.append(nodesDict)
            adjMatrix_list.append(adjMatrix)

        self.saveOutput(nodesDict_list, adjMatrix_list)
        return nodesDict_list, adjMatrix_list


    def sampleNodesByHistory(self, currentNode, nodesDict, neighbors, numNodesToConnect, coorpBaseProb = 0.1):
        nonNeighbors = [nodeID for nodeID in nodesDict.keys() if nodeID!= currentNode and nodeID not in neighbors]
        nodesCoopProb = {nodeID: 1 - nodesDict[nodeID].getDefectPercent() for nodeID in nonNeighbors}

        nodesSampleSpace = list(nodesCoopProb.keys())
        random.shuffle(nodesSampleSpace)

        nodesToConnect = []
        for nodeID in nodesSampleSpace:
            selectProb = max(nodesCoopProb[nodeID], coorpBaseProb)
            select = np.random.choice([0, 1], 1, p=[1-selectProb, selectProb])[0]
            if select:
                nodesToConnect.append(nodeID)
            if len(nodesToConnect) >= numNodesToConnect:
                break

        return nodesToConnect


    def simulateByStatusHistory(self, nodesDict, adjMatrix, strategy):
        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)  # adding initial structure
        numNodes = len(adjMatrix[0])

        for iterID in tqdm(range(self.numIterations)):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            brokenEdges = []
            nodesToFormNewEdge = []

            for i in range(0, numNodes):
                nodesDict[i].updateStatusHistory(nodesDict[i].status)

                for j in range(0, i):
                    if adjMatrix[i][j]:  # there is an edge
                        if nodesDict[i].status == 1 or nodesDict[j].status == 1:
                            brokenEdges.append((i, j))
                            nodesToFormNewEdge.append(i) if nodesDict[j].status == 1 else nodesToFormNewEdge.append(j)

                        # Update node payoff history
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j], iterID)

            adjMatrixBeforeChanging = adjMatrix.copy()
            nodesDictOld = nodesDict.copy()  # create a copy for reference

            # get edges to form:
            nodesToConnectDict = {}
            for nodeID in set(nodesToFormNewEdge):
                neighbors = [node for node in range(numNodes) if adjMatrix[nodeID][node]]
                numNodesToConnect = nodesToFormNewEdge.count(nodeID)
                nodesToConnect = self.sampleNodesByHistory(nodeID, nodesDict, neighbors, numNodesToConnect)
                nodesToConnectDict[nodeID] = nodesToConnect

            # break edges:
            for edge in brokenEdges:
                stub1, stub2 = edge
                adjMatrix[stub1][stub2] = 0
                adjMatrix[stub2][stub1] = 0

            # form new edges
            for nodeID in nodesToConnectDict.keys():
                nodesToConnect = nodesToConnectDict[nodeID]
                for node in nodesToConnect:
                    adjMatrix[nodeID][node] = 1
                    adjMatrix[node][nodeID] = 1

            # update nodes status
            for nodeID in nodesDict.keys():
                if strategy == 3:
                    nodesDict[nodeID] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[nodeID])

                elif strategy == 4:
                    nodeiNeighbors = [node for node, isNeighbor in zip(nodesDictOld.values(), adjMatrixBeforeChanging[nodeID]) if isNeighbor]
                    nodesDict[nodeID] = self.updateNodesStatusBasedOnNeighborsInfo(nodesDictOld[nodeID], nodeiNeighbors)

            nodesDict_list.append(nodesDict)
            adjMatrix_list.append(adjMatrix)

        self.saveOutput(nodesDict_list, adjMatrix_list)
        return nodesDict_list, adjMatrix_list

    def getNodesToAvoidWithMemory(self, nodesToAvoidInfo, nodeIID, nodeJID, numPunishRounds):
        # node i: node to form new edge
        # node j: node that just defected/ need to be avoided for the next n rounds

        currentNodesToAvoidForI = nodesToAvoidInfo.get(nodeIID, 'noNodesYet')  # a dict or 'noNodesYet'

        if currentNodesToAvoidForI == 'noNodesYet':
            # node i has no nodes to punish yet
            nodesToAvoidInfo[nodeIID] = {nodeJID: numPunishRounds}
        else:
            # node i has some nodes to punish
            timeForIToAvoidJ = currentNodesToAvoidForI.get(nodeJID, 0)  # node i punished j or not
            timeForIToAvoidJ += numPunishRounds
            nodesToAvoidInfo[nodeIID][nodeJID] = timeForIToAvoidJ

        return nodesToAvoidInfo

    def updateNodesStatusBasedOnLastPayoff(self, node):
        non_empty = 0
        for iteractions in node.lastPayoff:
            if len(iteractions) > 0:
                non_empty += 1
        if non_empty < 2:
            return node

        ## average of last iteration
        last_iteration = -1
        for i in range(len(node.lastPayoff) - 1, -1, -1):
            if len(node.lastPayoff[i]) > 0:
                last_iteration = i
                break

        last_avg = np.mean(node.lastPayoff[last_iteration])

        ## average of second last iteration
        second_last_interation = -1
        for i in range(last_iteration - 1, -1, -1):
            if len(node.lastPayoff[i]) > 0:
                second_last_interation = i
                break

        second_last_avg = np.mean(node.lastPayoff[second_last_interation])

        if last_avg < second_last_avg:
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)
        return node

    def updateNodesStatusBasedOnNeighborsInfo(self, node, neighborsList):
        if len(neighborsList) == 0:
            return node

        # average of last iteration for current node
        last_iteration = -1
        for i in range(len(node.lastPayoff) - 1, -1, -1):
            if len(node.lastPayoff[i]) > 0:
                last_iteration = i
                break

        nodeLastAvg = np.mean(node.lastPayoff[last_iteration])

        # average of last iteration for neighbors
        neighborsLastPayoff = []
        for neighbor in neighborsList:
            last_iteration = -1
            for i in range(len(neighbor.lastPayoff) - 1, -1, -1):
                if len(neighbor.lastPayoff[i]) > 0:
                    last_iteration = i
                    break

            last_avg = np.mean(neighbor.lastPayoff[last_iteration])
            neighborsLastPayoff.append(last_avg)

        neighborsMean = np.mean(neighborsLastPayoff)

        # change status based on comparison
        if nodeLastAvg < neighborsMean:
            neighborDefectCount = np.sum([neighbor.status for neighbor in neighborsList])
            newStatus = 1 if neighborDefectCount > len(neighborsList)/2 else 0 # conform to neighbors
            node.updateStatus(newStatus)

        return node

    def updateWithGamePayoff(self, node1, node2, time_stamp):
        newPayoff1, newPayoff2 = self.calcPayoff(node1.status, node2.status)
        node1.updatePayoff(newPayoff1, time_stamp)
        node2.updatePayoff(newPayoff2, time_stamp)
        return node1, node2

    def saveOutput(self, nodesDict_list, adjMatrix_list):
        # save nodes dict
        path = os.path.join(self.savePath, 'nodesDict_{}.pickle'.format(self.fileName))
        pklFile = open(path, "wb")
        pickle.dump(nodesDict_list, pklFile)
        pklFile.close()

        # save adjmatrix
        path = os.path.join(self.savePath, 'adjMat_{}.pickle'.format(self.fileName))
        pklFile = open(path, "wb")
        pickle.dump(adjMatrix_list, pklFile)
        pklFile.close()
