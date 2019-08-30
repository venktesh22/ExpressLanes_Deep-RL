#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:16:30 2019

@author: vpandey
"""

from network import Network
import numpy as np

def determineLaneChoice(net, stochastic, logitTheta=6):
    """
    Thus function returns the probability of choosing ML (or exit if exit diverge) by travelers in 
    each group in each diverge cell
    
    Parameters:
        network net
        stochastic: if True, then it uses multinomial/binary logit to
        find the probability of choosing the routes. If false, then it 
        chooses the first link of the path that maximizes the utility
    
    Returns a dictionary:
        diverge cell --> {groupID : probability of choosing ML}
    
    
    """
    #initialize
    routeChoiceProb = {}
    for cellID in net.divergeCellsByID:
        cell = net.divergeCellsByID[cellID]
        tempDict={}
        for group in net.groupsByID:
            tempDict[group]= 0.0
        routeChoiceProb[cell]=tempDict
    
    #if a group is exiting before the first exit from ML, then do nothing (probabilities are zero)
    #only if group's destination is downstream of first exit from ML, proceed
    for cellID in net.divergeCellsByID:
        cell = net.divergeCellsByID[cellID]
        if cell.endCell==None:
            #it is a diverge that leads to an exit..set prob of all groups exiting here to 1
            exitDest=-1
            for conn in cell.nextConnectors:
                if conn.toCell.cellClass=="OUTLINK":
                    exitDest= round(conn.toCell.id/100)%100 #2nd and 3rd digit out of 5 digits
            for dest in net.groupIDsByDest:
                if dest==exitDest:
                    for gId in net.groupIDsByDest[dest]:
                        routeChoiceProb[cell][gId]=1.0
            continue
        for dest in net.groupIDsByDest:
            if int(cell.endCell.id/10000) < dest:
                #only then is this group allowed to compare different routes
                pathsTT, pathsToll = computePathTTandToll(net, cell.pathToEndCell)
                for gId in net.groupIDsByDest[dest]:
                    vot = net.groupsByID[gId].vot
                    pathsUtil = -1*(np.array(pathsTT)*vot+np.array(pathsToll))
                    #print("cell %d has pathUtils %s" % (cellID, pathsUtil))
                    if not stochastic:
                        pathArgMax = np.argmax(pathsUtil)
                        maxPathFirstCellID = cell.pathToEndCell[pathArgMax][1] #technically second cell
                        if net.allCellsByID[maxPathFirstCellID].cellClass == "ONRAMP" or net.allCellsByID[maxPathFirstCellID].cellClass == "HOT":
                            probML=1
                            routeChoiceProb[cell][gId]=probML
                    else:
                        #stochastic choice...compute probabilities using Logit
#                        print("===Cell is ",cell,"=====")
#                        print("Path utils for group ",gId," for dest ", dest," are",pathsUtil)
                        expPathUtils = np.exp(logitTheta*pathsUtil)
                        if sum(expPathUtils)>0:
                            prob = expPathUtils/sum(expPathUtils)
                        else:
                            print("\n\n\nERROR in path probabilities.....\n\n\n")
                        #print("prob is ",prob)
                        probML=0
                        index=0
                        for path in cell.pathToEndCell:
                            firstCellID = path[1]
                            if net.allCellsByID[firstCellID].cellClass == "ONRAMP" or net.allCellsByID[firstCellID].cellClass == "HOT":
                                probML= probML+prob[index]
                            index=index+1
                        routeChoiceProb[cell][gId]=probML
#                print("Route choice prob matrix is ", routeChoiceProb)
    return routeChoiceProb
    
def computePathTTandToll(network, paths):
    pathsTT=[]
    pathsToll=[]
    for path in paths:
        pathTT=0.0
        pathToll=0.0
        for cellID in path:
            cell= network.allCellsByID[cellID]
            pathTT = pathTT + cell.getCurrentTravelTime()
            pathToll = pathToll + cell.getToll()
        pathsTT.append(pathTT)
        pathsToll.append(pathToll)
    return pathsTT, pathsToll