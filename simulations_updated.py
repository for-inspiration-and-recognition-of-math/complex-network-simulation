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
                return 1, 1
            elif status1 == 0 and status2 == 1:
                return -1, 2
            elif status1 == 1 and status2 == 0:
                return 2, -1
            else:
                return 0, 0

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
                return 0, 0

class Node:
    def __init__(self, status):
        self.status = status # 0 = cooperate, 1 = defect
        self.lastPayoff = []

    def updatePayoff(self, newPayoff, time_stamp):
        self.lastPayoff[time_stamp].append(newPayoff) # [[2, 0, 3], [1, 2], [], [3]]

    def updateStatus(self, newStatus):
        self.status = newStatus
        
    def initalizeLastPayoffList(self, numIterations):
        self.lastPayoff = [ [] for i in range (numIterations) ]


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
        for node in nodesDict.values():
            node.initalizeLastPayoffList(self.numIterations)
            
        if self.strategy == 0:
            return self.simulation_simplest(nodesDict, adjMatrix)
        else:
            return self.simulation_unweighted(nodesDict, adjMatrix, self.strategy)

    def simulation_simplest(self, nodesDict, adjMatrix, pInteract = .1):
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
                    if adjMatrix[i][j] and random.uniform(0, 1) < pInteract: # there is an edge then interact with prob = p
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

    def linkToNewNode(self, adjMatrix, nodeID, excludingNodesID, numNewEdgesToForm):
        otherNodesID = [otherNodeID for otherNodeID, isNeighbor in enumerate(adjMatrix[nodeID])
                        if not isNeighbor and otherNodeID not in excludingNodesID]
        if len(otherNodesID) == 0:
            return adjMatrix

        numNewEdgesToForm = numNewEdgesToForm if len(otherNodesID) >= numNewEdgesToForm else len(otherNodesID)
        newNodesToLink = np.random.choice(otherNodesID, numNewEdgesToForm, replace=False)
        for newNodeID in newNodesToLink:
            adjMatrix[nodeID][newNodeID] = 1
            adjMatrix[newNodeID][nodeID] = 1

        return adjMatrix

    def simulation_unweighted(self, nodesDict, adjMatrix, type):
        nodesDict_list, adjMatrix_list = [], []
        nodesDict_list.append(nodesDict)
        adjMatrix_list.append(adjMatrix)  # adding initial structure
        numNodes = len(adjMatrix[0])

        for iterID in tqdm(range(self.numIterations)):
            adjMatrix = deepcopy(adjMatrix_list[-1])
            nodesDict = deepcopy(nodesDict_list[-1])

            brokenEdges = []
            nodesToAvoidInfo = {} # {nodeWhoFormsNewEdge: [nodesToAvoid]}

            for i in range(0, numNodes):
                for j in range(0, i):
                    if adjMatrix[i][j]:  # there is an edge
                        if type != 3: # break/form new edges
                            if nodesDict[i].status == 1 or nodesDict[j].status == 1:
                                brokenEdges.append((i, j))
                                if nodesDict[j].status == 1: # form new edges for i
                                    currentNodesToAvoidForI = nodesToAvoidInfo.get(i, 'noNodesYet')
                                    newNodesToAvoidForI = currentNodesToAvoidForI + [j] if currentNodesToAvoidForI is not 'noNodesYet' else [j]
                                    nodesToAvoidInfo[i] = newNodesToAvoidForI

                                else:
                                    currentNodesToAvoidForJ = nodesToAvoidInfo.get(j, 'noNodesYet')
                                    newNodesToAvoidForJ = currentNodesToAvoidForJ + [i] if currentNodesToAvoidForJ is not 'noNodesYet' else [i]
                                    nodesToAvoidInfo[j] = newNodesToAvoidForJ

                        # Update node payoff history
                        nodesDict[i], nodesDict[j] = self.updateWithGamePayoff(nodesDict[i], nodesDict[j], iterID)

            # break edges:
            for edge in brokenEdges:
                stub1, stub2 = edge
                adjMatrix[stub1][stub2] = 0
                adjMatrix[stub2][stub1] = 0

            # form new edges
            for nodeIDToCreateNewEdge in nodesToAvoidInfo.keys():
                # TODO: ordering of nodes still matters a bit?
                nodesJustInteracted = nodesToAvoidInfo[nodeIDToCreateNewEdge]
                excludingNodesID = nodesJustInteracted + [nodeIDToCreateNewEdge] # no edges with nodes just interacted, no self-edge
                numNewEdges = len(nodesJustInteracted)
                adjMatrix = self.linkToNewNode(adjMatrix, nodeIDToCreateNewEdge, excludingNodesID, numNewEdges)

            # update nodes status
            for nodeID in nodesDict.keys():
                if type == 1:
                    nodesDict[nodeID] = self.updateNodesStatusBasedOnLastPayoff(nodesDict[nodeID])

                elif type == 2 or type == 3:
                    nodeiNeighbors = [node for node, isNeighbor in zip(nodesDict.values(), adjMatrix[nodeID]) if isNeighbor]
                    nodesDict[nodeID] = self.updateNodesStatusBasedOnAvgPayoff(nodesDict[nodeID], nodeiNeighbors)

            nodesDict_list.append(nodesDict)
            adjMatrix_list.append(adjMatrix)

        self.saveOutput(nodesDict_list, adjMatrix_list)
        return nodesDict_list, adjMatrix_list


    def updateNodesStatusBasedOnAvgPayoff(self, node, neighborsList):
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
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)

        return node

    def updateNodesStatusBasedOnLastPayoff(self, node):
        non_empty = 0
        for iteractions in node.lastPayoff:
            if len(iteractions) > 0:
                non_empty += 1
        if non_empty < 2:
            return node
 
        ## average of last iteration
        last_iteration = -1
        for i in range (len(node.lastPayoff) - 1, -1, -1):
            if len(node.lastPayoff[i]) > 0:
                last_iteration = i
                break
            
        last_avg = np.mean(node.lastPayoff[last_iteration])
        
        ## average of second last iteration
        second_last_interation = -1
        for i in range (last_iteration - 1, -1, -1):
            if len(node.lastPayoff[i]) > 0:
                second_last_interation = i
                break
        
        second_last_avg = np.mean(node.lastPayoff[second_last_interation])
        
        if last_avg < second_last_avg:
            newStatus = 1 if node.status == 0 else 0
            node.updateStatus(newStatus)
        return node

    def updateWithGamePayoff(self, node1, node2, time_stamp):
        newPayoff1, newPayoff2 = self.calcPayoff(node1.status, node2.status)
        node1.updatePayoff(newPayoff1, time_stamp)
        node2.updatePayoff(newPayoff2, time_stamp)
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