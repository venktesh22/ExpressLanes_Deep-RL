#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 07:08:54 2019
This file defines the network with its cells and all its connections
A snapshot of the network represents the state of the RL problem
@author: vpandey
"""
##import os
#import sys
##path = os.getcwd()
##sys.path.insert(0,path+'/gym-meme/gym_meme/envs/components/')
#sys.path.insert(0,'/Users/vpandey/Box Sync/UT Acads/PhD Dissertation/Modeling/MEME_DRL/gym-meme/gym_meme/envs/components/')

from cell import Cell
from group import Group
from cellConnector import CellConnector
from node import Node
from link import Link
import reader as rd
import numpy as np
import pandas as pd

class Network:
    def __init__(self, netname, seed=1000):
        print("=====READING THE NETWORK======")
        self.allNodesByID, self.allLinksByID, self.minSpeedLimit= rd.readLinkFile(netname)
        self.votVals = rd.readVOTFile(netname)
        self.netName, self.simTime, self.timestep, self.tollUpdateStep, self.dFactor, self.minSpeedHOT, self.routeType, self.laneStoch, self.stddev_demand= rd.readParametersFile(netname)
        self.netName = netname
        self.destinations = rd.readDemandFile(netname).DestNode.unique();
        self.deltaX = round((self.minSpeedLimit*self.timestep)/36)/100.0 #length of each cell
        self.netSeed = seed

        self.tollMin = 0.1 #units $ 0.01 for DESE
        self.tollMax = 4.0 # units $ 0.6 for DESE
        
        if netname=='DESEw1D':
            self.tollMin=0.01 #we do this to be consistent with Pandey and Boyles (2018) assumptions
            self.tollMax=0.6
        self.currentTime = 0 #this is the current time in the network...
                             #starts with 0, updates by self.timestep seconds everytime we propagate flow
        
        self.revenue=0
        self.TSTTmeasure=0  #doesn't measure the actual TSTT but for one step counts how many vehicles
        self.vehiclesExited = 0
        self.jah_Static = -1000 #stores the max of difference between (no of vehicles in GPL - vehs in ML)
        self.jah_Statistic2 = 0 #stores the sum of (difference between avg density of GPL and ML) over all timesteps in each toll update step
        self.tdjah_nstat = [] #stores the normalized jah stat for each time
        
        self.sumSqMLViolation = 0 #sum of sqaure of difference between speed and minspeedrequired if speed<minspeedLimit
        self.countMLViolation = 0 #count how many cells for how many timesteps are violated or have speed<minspeedLimit
        self.observationLinksOrdered=[]
        self.tollProfile=[] #the toll profile holds the tolls charged from beginning to the end, and gets reset when network is reset
        
        self.createNetwork()
        self.assignDemandToSourceCells(netname)
        self.readLinksObservedFromFile = False
        if netname=='DESEw1D' or netname=='LBJ':
            self.readLinksObservedFromFile = False
        self.determineObservationLinks()
        self.printNetwork()
        #seed(1)
        
        self.collectTSdiagramInfo = True #whether or not to store density every run...memory consuming!
    
    def createNetwork(self):
        self.defineGroups()
        self.createCells()
        self.createCellConnections()
        self.createDivergeCellRoutes()

    """
    Defines the individual groups
    """
    def defineGroups(self):
        id=0
        self.groupsByID={}
        self.groupIDsByVOT={}
        for vot in self.votVals:
            self.groupIDsByVOT[vot]=[]
        self.groupIDsByDest={}
        for dest in self.destinations:
            self.groupIDsByDest[dest]=[]
        for dest in self.destinations:
            for vot in self.votVals:
                g= Group(id,vot, self.allNodesByID[dest])
                self.groupsByID[id]=g
                self.groupIDsByVOT[vot].append(id)
                self.groupIDsByDest[dest].append(id)
                id = id+1
    """
    Creates all cells
    """
    def createCells(self):
        self.allCellsByID = {}
        self.mergeCellsByID = {}
        self.divergeCellsByID = {}
        self.tollCells = [] #we want toll cells to be an ordered list
        self.HOTCellIDs = [] # we want ordered
        self.GPCellIDs = []
        
        cellType=""; cellID=0;
        print("No of cells created for each link:")
        for lID in self.allLinksByID:
            l=self.allLinksByID[lID]
            #rounding right now, so only link lengths which are multiple of deltaX allowed
            l.noOfCells = round(l.length/self.deltaX)
            noIncomingLinks = len(l.fromNode.incoming)
            noOutgoingLinks = len(l.toNode.outgoing)
            linkCells = []
            for i in range(l.noOfCells):
                cellID=lID*100+i+1
                isTolled= False
                if i==0:
                    #first cell of link can be merge, source, diverge, or ordinary
                    if noIncomingLinks==2:
                        cellType="merge"
                    elif noIncomingLinks==0:
                        cellType="source"
                    elif noOutgoingLinks==2 and l.noOfCells==1:
                        cellType="diverge"
                    else:
                        cellType="ordinary"
                    if l.linkClass=="ONRAMP" or (l.linkClass=="HOT" and len(l.fromNode.outgoing)==2):
                        isTolled= True
                elif i== l.noOfCells-1:
                    #provided there are more than one cell, the last cell could be sink or diverge
                    if noOutgoingLinks==2:
                        cellType="diverge"
                    elif noOutgoingLinks==0:
                        cellType="sink"
                    else:
                        cellType="ordinary"
                else:
                    #if more than one cells and not the last cell, then it is ordinary
                    cellType="ordinary"
                cell = Cell(cellID, l,  cellType, l.linkClass, self.deltaX, l.FFS, l.BWS,
                            (l.capacity), round(l.jamDensity*self.deltaX),
                            isTolled, self.groupsByID)
                linkCells.append(cell)
                self.allCellsByID[cellID]=cell
                if cellType=="merge":
                    self.mergeCellsByID[cellID]=cell
                if cellType=="diverge":
                    self.divergeCellsByID[cellID]=cell
                if isTolled:
                    self.tollCells.append(cell)
                    cell.toll=self.tollMin #initialize
                
                if l.linkClass=="HOT":
                    self.HOTCellIDs.append(cellID)
                elif l.linkClass=="GP":
                    self.GPCellIDs.append(cellID)
            l.allCells= linkCells
            print("===",l, len(linkCells))
        self.HOTCellIDs = np.array(self.HOTCellIDs)
        self.HOTCellIDs.sort()
        self.GPCellIDs = np.array(self.GPCellIDs)
        self.GPCellIDs.sort()
        
        self.allHOTTimestampedDensities= pd.DataFrame(index=self.HOTCellIDs)
        self.allGPTimestampedDensities= pd.DataFrame(index=self.GPCellIDs)
    
    def createCellConnections(self):
        self.cellConnectorsByCellIDs = {} #fromCellID --> {toCellID, CellConnector}
        self.allCellConnectors=[]
        #connect all cells within a link; these are ordinary connectors
        for lID in self.allLinksByID:
            allCells = self.allLinksByID[lID].allCells
            for i in range(len(allCells)-1):
                fromCell = allCells[i]
                toCell = allCells[i+1]
                conn = CellConnector(fromCell, toCell,"ordinary")
                self.allCellConnectors.append(conn)
                
                fromCell.nextConnectors.append(conn)
                toCell.prevConnectors.append(conn)
                
                tempDict={}
                tempDict[toCell.id]=conn
                self.cellConnectorsByCellIDs[fromCell.id]=tempDict
        #connect cells from one link to next
        for nodeID in self.allNodesByID:
            incomingL = self.allNodesByID[nodeID].incoming
            outgoingL = self.allNodesByID[nodeID].outgoing
            if len(incomingL)==1 and len(outgoingL)==2:
                #diverge node
                incomingL_lastcell = incomingL[0].allCells[incomingL[0].noOfCells-1]
                outgoingL1_firstcell = outgoingL[0].allCells[0]
                outgoingL2_firstcell = outgoingL[1].allCells[0]
                conn1 = CellConnector(incomingL_lastcell, outgoingL1_firstcell,"diverge")
                conn2 = CellConnector(incomingL_lastcell, outgoingL2_firstcell,"diverge")
                self.allCellConnectors.append(conn1)
                self.allCellConnectors.append(conn2)
                
                incomingL_lastcell.nextConnectors.append(conn1)
                incomingL_lastcell.nextConnectors.append(conn2)
                outgoingL1_firstcell.prevConnectors.append(conn1)
                outgoingL2_firstcell.prevConnectors.append(conn2)
                
                tempDict1={}
                tempDict1[outgoingL1_firstcell.id]=conn1
                tempDict1[outgoingL2_firstcell.id]=conn2
                self.cellConnectorsByCellIDs[incomingL_lastcell.id]=tempDict1
            elif len(incomingL)==2 and len(outgoingL)==1:
                #merge node
                incomingL1_lastcell = incomingL[0].allCells[incomingL[0].noOfCells-1]
                incomingL2_lastcell = incomingL[1].allCells[incomingL[1].noOfCells-1]
                outgoingL_firstcell = outgoingL[0].allCells[0]
                conn1 = CellConnector(incomingL1_lastcell, outgoingL_firstcell,"merge")
                conn2 = CellConnector(incomingL2_lastcell, outgoingL_firstcell,"merge")
                self.allCellConnectors.append(conn1)
                self.allCellConnectors.append(conn2)
                
                incomingL1_lastcell.nextConnectors.append(conn1)
                incomingL2_lastcell.nextConnectors.append(conn2)
                outgoingL_firstcell.prevConnectors.append(conn1)
                outgoingL_firstcell.prevConnectors.append(conn2)
                
                tempDict1={}
                tempDict1[outgoingL_firstcell.id]=conn1
                self.cellConnectorsByCellIDs[incomingL1_lastcell.id]=tempDict1
                tempDict2={}
                tempDict2[outgoingL_firstcell.id]=conn2
                self.cellConnectorsByCellIDs[incomingL2_lastcell.id]=tempDict2
            elif len(incomingL)==1 and len(outgoingL)==1:
                #regular node (not source or sink)
                incomingL_lastcell = incomingL[0].allCells[incomingL[0].noOfCells-1]
                outgoingL_firstcell = outgoingL[0].allCells[0]
                conn = CellConnector(incomingL_lastcell, outgoingL_firstcell,"ordinary")
                self.allCellConnectors.append(conn)
                
                incomingL_lastcell.nextConnectors.append(conn)
                outgoingL_firstcell.prevConnectors.append(conn)
                
                tempDict1={}
                tempDict1[outgoingL_firstcell.id]=conn
                self.cellConnectorsByCellIDs[incomingL_lastcell.id]=tempDict1

    #---defines the routes connecting each diverge cell to the end of first exit from ML if entering now
    def createDivergeCellRoutes(self):
        for cellID in self.divergeCellsByID:
            cell = self.divergeCellsByID[cellID]
            if cell.nextConnectors[0].toCell.cellClass=="OUTLINK" or cell.nextConnectors[1].toCell.cellClass=="OUTLINK":
                continue
            endCell = self.determineEndCell(cell.link.toNode)
            #print('end cell for diverge cell %s is %s' % (cell,endCell))
            paths = self.determinePaths(cell, endCell)
            cell.endCell = endCell
            if self.routeType=="decisionRoute" or self.routeType=="DR":
                cell.pathToEndCell = paths
            elif self.routeType.lower()=="binary":
                #logit routes consist of paths along HOT or GP alone...
                validPaths=[] #all oaths that only take HOT or only take GP
                for path in paths:
                    secondCell = self.allCellsByID[path[1]]
                    lastCell = self.allCellsByID[path[-1]]
                    if secondCell.cellClass=="GP" and lastCell.cellClass=="GP":
                        validPaths.append(path)
                    elif secondCell.cellClass=="ONRAMP" and lastCell.cellClass=="OFFRAMP":
                        validPaths.append(path)
                    elif secondCell.cellClass=="OFFRAMP" and lastCell.cellClass=="GP":
                        validPaths.append(path)
                    elif secondCell.cellClass=="HOT" and lastCell.cellClass=="OFFRAMP":
                        validPaths.append(path)
                paths=validPaths
                cell.pathToEndCell = paths
            #print("path is", paths)
    
    def determineEndCell(self, divergeNode):
        finalNodeID = divergeNode.id
        testClass = "OFFRAMP" #we look for first offramps if entered now
        isHOTCell = False
        if divergeNode.incoming[0].linkClass == "HOT":
            isHOTCell = True
        breakLoop = False
        #loop in topological order
        for i in range(divergeNode.id, len(self.allNodesByID),1):
            tempNode = self.allNodesByID[i]
            for l in tempNode.outgoing:
                if l.linkClass==testClass:
                    if isHOTCell:
                        #if first link is offramp, then continue
                        isHOTCell= False
                        continue
                    finalNodeID = l.toNode.id
                    breakLoop= True
                    break
            if breakLoop:
                break
        return self.allCellsByID[self.allNodesByID[finalNodeID].outgoing[0].id*100+1]
    
    def determinePaths(self, startCell, endCell):
        allPaths = []
        tmp = []
        self.findAllPathsAt(startCell, endCell, allPaths, tmp)
        return allPaths
    
    #recursive function that finds all paths from cell o to cell d and 
    #stores the enumeration of nodes in tmp
    def findAllPathsAt(self, o, d, allPaths, tmp):
#        if len(o.nextConnectors)==0 or o.id>=d.id:
        if o.id>=d.id:
            allPaths.append(tmp)
            return
        tmp.append(o.id)
        for conn in o.nextConnectors:
            nextCell = conn.toCell
            tmp2=[]
            for i in tmp:
                tmp2.append(i)
            if nextCell.id>d.id:
                break
            self.findAllPathsAt(nextCell, d, allPaths, tmp2)
    
    
    def assignDemandToSourceCells(self, netname):
        np.random.seed = self.netSeed
        demandData = rd.readDemandFile(netname)
        for origin in demandData.OriginNode.unique():
            for dest in demandData.DestNode.unique():
                #find the source cell
                cID= self.allNodesByID[origin].outgoing[0].id*100+1
                sourceCell = self.allCellsByID[cID]
                #determine subselect data with this origin and dest
                subsetData = demandData.loc[(demandData['OriginNode']==origin) & (demandData['DestNode']==dest)]
                for index, row in subsetData.iterrows():
                    startTime = row['StartTime(sec)']
                    endTime = row['EndTime(sec)']
                    demand= row['Demand(veh/hr)']
                    for t in range(startTime,endTime,self.timestep):
                        tempDict={}
                        if t in sourceCell.sourceDemand:
                            tempDict=sourceCell.sourceDemand[t]
                        for groupID in self.groupIDsByDest[dest]:
#                            meanDemand = demand* self.dFactor* self.votVals[self.groupsByID[groupID].vot]
#                            sampledDemand = np.random.normal(meanDemand, self.stddev_demand)
#                            if sampledDemand<0.0:
#                                sampledDemand=0.0
                            tempDict[groupID]= demand* self.dFactor* self.votVals[self.groupsByID[groupID].vot]
                        sourceCell.sourceDemand[t]=tempDict
                if endTime< self.simTime:
                    for t in range(endTime,self.simTime,self.timestep):
                        tempDict={}
                        if t in sourceCell.sourceDemand:
                            tempDict=sourceCell.sourceDemand[t]
                        for groupID in self.groupIDsByDest[dest]:
                            tempDict[groupID]=0.0
                        sourceCell.sourceDemand[t]=tempDict
#                print("===Source cell==",sourceCell)
#                print(sourceCell.sourceDemand)
                        
    def determineObservationLinks(self):
        
        if self.readLinksObservedFromFile:
            linkIDs=rd.readObservedLinksFile(self.netName)
            for linkID in linkIDs:
                l= self.allLinksByID[linkID]
                self.observationLinksOrdered.append(l) 
        else:
            for linkID in self.allLinksByID:
                l= self.allLinksByID[linkID]
                if l.linkClass=="GP" or l.linkClass=="HOT":
                    self.observationLinksOrdered.append(l)
    
    def resetNetwork(self):
        #reset time
        self.currentTime = 0
        self.revenue=0
        self.TSTTmeasure=0  #doesn't measure the actual TSTT but for one step counts how many vehicles
        self.vehiclesExited = 0
        self.tollProfile=[]
        self.jah_Static = -1000
        self.jah_Statistic2 = 0
        self.tdjah_nstat = []
        self.sumSqMLViolation = 0
        self.countMLViolation = 0
        #reset no of vehicles
        for cellID in self.allCellsByID:
            cell = self.allCellsByID[cellID]
            for groupID in self.groupsByID:
                cell.vehiclesByGroup[groupID]=0.0
                for conn in cell.nextConnectors:
                    conn.vehiclesByGroup[groupID]=0.0
        
    
    def setToll(self, toll):
        index=0
        assert len(toll)==len(self.tollCells),"Dimension mismatch while setting tolls"
        tempTollVar=[] #used to avoid tensorflow printing and just get toll value
        for cell in self.tollCells:
            cell.toll= toll[index]
            tempTollVar.append(toll[index])
            index=index+1
        self.tollProfile.append(tempTollVar)
        
    def printNetwork(self):
        print("\nReading network completed...")
        print("No of links:", len(self.allLinksByID))
        print("No of nodes:", len(self.allNodesByID))
        print("Diverge cells", len(self.divergeCellsByID))
        print("toll cells", len(self.tollCells))
        print("merge cells", len(self.mergeCellsByID))
        print("ALl cell connectors")
        print(len(self.cellConnectorsByCellIDs))
#        for cellID in self.allCellsByID:
#            print(cellID, self.allCellsByID[cellID].nextConnectors, 
#                  self.allCellsByID[cellID].prevConnectors)
        for cellID in self.divergeCellsByID:
            print(cellID,"paths:", len(self.divergeCellsByID[cellID].pathToEndCell))
#            , self.divergeCellsByID[cellID].pathToEndCell)
        print("Links are being observed in this order:",self.observationLinksOrdered)
#        for cellID in self.allCellsByID:
#            print(cellID,self.allCellsByID[cellID].vehiclesByGroup)
    
    def printFlows(self):
        for cellID in self.allCellsByID:
            print(cellID," ===> ",self.allCellsByID[cellID].vehiclesByGroup)

    def getId(self):
        return self.netName
    
    def collectTimeSpaceDiagramInfo(self):
        HOTDensity=[]
        GPDensity=[]
        for cellID in self.HOTCellIDs:
            cell = self.allCellsByID[cellID]
            value= sum(cell.vehiclesByGroup.values())/cell.maxNoOfVeh
            if abs(value)<1e-5:
                value=0
            HOTDensity.append(value)
        for cellID in self.GPCellIDs:
            cell = self.allCellsByID[cellID]
            value= sum(cell.vehiclesByGroup.values())/cell.maxNoOfVeh
            if abs(value)<1e-5:
                value=0
            GPDensity.append(value)
#        HOTDensity = np.round(np.array(HOTDensity),2)
#        GPDensity = np.round(np.array(GPDensity),2)
        self.allHOTTimestampedDensities[str(self.currentTime)]= np.array(HOTDensity)
        self.allGPTimestampedDensities[str(self.currentTime)]=np.array(GPDensity)
    
    def printTimeSpaceDiagram(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        data=self.allHOTTimestampedDensities
        data = data.iloc[::-1]
        sns.heatmap(data, vmin=0, vmax=1, cmap="RdYlGn_r")
        plt.title("ML Time Space Diagram", fontsize=16)
        plt.xlabel("Time (sec)", fontsize=16)
        plt.ylabel("Cell ID", fontsize=16)
        plt.tight_layout()
        plt.show()
        
        data=self.allGPTimestampedDensities
        data = data.iloc[::-1]
        sns.heatmap(data, vmin=0, vmax=1, cmap="RdYlGn_r")
        plt.title("GPL Time Space Diagram", fontsize=16)
        plt.xlabel("Time (sec)", fontsize=16)
        plt.ylabel("Cell ID", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plottdJAH(self):
        import matplotlib.pyplot as plt
        print("===The maximum jah value is %f=======" % (max(self.tdjah_nstat)))
        plt.plot(self.tdjah_nstat)
        plt.xlabel('Time step (t)', fontsize=16)
        plt.ylabel(r'$\zeta(t)$', fontsize=16)
        plt.show()
        print("===The maximum jah value is %f=======" % (max(self.tdjah_nstat)))

if __name__=='__main__':
    #n=Network('SESEw1Dest')
    n=Network('Mopac')
    #n=Network('DESE')
    print(n.stddev_demand)
    print(n.laneStoch)
    print(n.groupIDsByVOT,n.groupIDsByDest)
    #print(n.allNodesByID, n.netName, n.simTime, n.timestep)