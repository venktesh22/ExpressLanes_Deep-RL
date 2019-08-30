#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:53:06 2019
This function propagates flow in the network using multiclass CTM for one timestep
@author: vpandey
"""
from network import Network
import lane_choice
import sys
import numpy as np
#sys.path.insert(0,'/Users/vpandey/Box Sync/UT Acads/PhD Dissertation/Modeling/MEME_DRL/gym-meme/gym_meme/envs/')

def propogateFlow(network):
    np.random.seed= network.netSeed
    #=================================================#
    #==update flow of every ordinary cell connector===#
    #=================================================#
    for cellConn in network.allCellConnectors:
        if cellConn.type=="ordinary":
            backWaveRatio = cellConn.toCell.backWaveSpeed/cellConn.toCell.freeFlowSpeed
            currentTotalVeh = sum(cellConn.fromCell.vehiclesByGroup.values())
            nextTotalVeh = sum(cellConn.toCell.vehiclesByGroup.values())
            flow = min( cellConn.fromCell.capacity*network.timestep/3600.0 ,
                       min (cellConn.toCell.capacity*network.timestep/3600.0,
                              min(currentTotalVeh, backWaveRatio*
                                  (cellConn.toCell.maxNoOfVeh-nextTotalVeh))))
            if currentTotalVeh>1E-8:
                for groupID in network.groupsByID:
                    cellConn.vehiclesByGroup[groupID] = flow*cellConn.fromCell.vehiclesByGroup[groupID]/currentTotalVeh
            else:
                for groupID in network.groupsByID:
                    cellConn.vehiclesByGroup[groupID] = 0.0
    
    #=================================================#
    #==update flow of every diverge cell connector====#
    #=================================================#
    laneChoices = lane_choice.determineLaneChoice(network, network.laneStoch)
    #print("\nProb of choosing ML are=", laneChoices,"\n=======\n")
    for cellID in network.divergeCellsByID:
        cell= network.divergeCellsByID[cellID]
        mlLaneChoiceProbByGroup = laneChoices[cell]
#        print("ML probability for cell %s is %s" % (cellID, mlLaneChoiceProbByGroup))
        #access both its connectors
        conn1= []
        for key, value in network.cellConnectorsByCellIDs[cellID].items():
            conn1.append(value)
            if value.toCell.cellClass == "ONRAMP" or value.toCell.cellClass == "HOT" or value.toCell.cellClass == "OUTLINK":
                mlCell = value.toCell
            else:
                gplCell = value.toCell
        
        if sum(cell.vehiclesByGroup.values())>0.0:
            
            #find receiving flow
            backWaveRatio = mlCell.backWaveSpeed/mlCell.freeFlowSpeed
            receivingFlowML= min(mlCell.capacity*network.timestep/3600.0,
                                backWaveRatio*
                              (mlCell.maxNoOfVeh-sum(mlCell.vehiclesByGroup.values())))
            backWaveRatio = gplCell.backWaveSpeed/gplCell.freeFlowSpeed
            receivingFlowGPL= min(gplCell.capacity*network.timestep/3600.0,
                                backWaveRatio*
                              (gplCell.maxNoOfVeh-sum(gplCell.vehiclesByGroup.values())))
            #find sending flows using given probabilities of choosing ML (so use probability as proportions which might be an assumption)
            sendingFlowML=0.0
            sendingFlowGPL=0.0
            totalFlowWillingToGoToML = 0.0 #this is same as sendingFlow if capacity term doesn't dominate
            totalFlowWillingToGoToGPL = 0.0
            for groupID in network.groupsByID:
                sendingFlowML = sendingFlowML + cell.vehiclesByGroup[groupID]*mlLaneChoiceProbByGroup[groupID]
                sendingFlowGPL = sendingFlowGPL + cell.vehiclesByGroup[groupID]*(1-mlLaneChoiceProbByGroup[groupID])
            totalFlowWillingToGoToML = sendingFlowML
            totalFlowWillingToGoToGPL = sendingFlowGPL
            sendingFlowML = min(sendingFlowML, cell.capacity*network.timestep/3600.0)
            sendingFlowGPL = min(sendingFlowGPL, cell.capacity*network.timestep/3600.0)
            
#            print(receivingFlowML, receivingFlowGPL, sendingFlowML, sendingFlowGPL)
            #evaluate Phi, proportion of vehicles that do to each downstream cell
            phi=0.0
            if sendingFlowML>0.0 and sendingFlowGPL>0.0:
                phi = min(1, min(receivingFlowML/sendingFlowML, receivingFlowGPL/sendingFlowGPL))
            elif sendingFlowML==0.0 and sendingFlowGPL>0.0:
                phi= min(1, receivingFlowGPL/sendingFlowGPL)
            elif sendingFlowML>0.0 and sendingFlowGPL==0.0:
                phi= min(1, receivingFlowML/sendingFlowML)
#            print("sendingFlowML %f, sendingFlowGPL %f, receivingFlowML %f, receivingFlowGPL %f capacity %f" %
#                  (sendingFlowML, sendingFlowGPL,receivingFlowML, receivingFlowGPL, cell.capacity*network.timestep/3600.0))
                
#            print('phi this time = %f' % (phi))
#            if phi<0.1:
#                print("A lane is very congested preventing other lanes at diverge to move. Investigate!")
            #evaluate the cell connector flow
            
            nextConnML = network.cellConnectorsByCellIDs[cell.id][mlCell.id]
            nextConnGPL = network.cellConnectorsByCellIDs[cell.id][gplCell.id]
            
            for groupID in network.groupsByID:
                p = mlLaneChoiceProbByGroup[groupID]
                if totalFlowWillingToGoToML<1E-8:
                    nextConnML.vehiclesByGroup[groupID] = 0.0
                else:
                    nextConnML.vehiclesByGroup[groupID] = (phi*sendingFlowML)*(p*cell.vehiclesByGroup[groupID]/totalFlowWillingToGoToML)
                if totalFlowWillingToGoToGPL<1E-8:
                    nextConnGPL.vehiclesByGroup[groupID] = 0.0
                else:
                    nextConnGPL.vehiclesByGroup[groupID] = phi*sendingFlowGPL*(1-p)*(cell.vehiclesByGroup[groupID]/totalFlowWillingToGoToGPL)
        else:
            for groupID in network.groupsByID:
                network.cellConnectorsByCellIDs[cell.id][mlCell.id].vehiclesByGroup[groupID] = 0.0
                network.cellConnectorsByCellIDs[cell.id][gplCell.id].vehiclesByGroup[groupID] = 0.0
        network.revenue = network.revenue + sum(network.cellConnectorsByCellIDs[cell.id][mlCell.id].vehiclesByGroup.values())*mlCell.toll
        #print("Diverge cell connector flows computed")
    #=================================================#
    #==update flow of every merge cell connector====#
    #=================================================#
    for cellID in network.mergeCellsByID:
        cell=network.mergeCellsByID[cellID]
        prevConn1=cell.prevConnectors[0]
        prevConn2=cell.prevConnectors[1]
        prevCell1 = prevConn1.fromCell
        prevCell2 = prevConn2.fromCell
        
        sendingFlow1= min( sum(prevCell1.vehiclesByGroup.values()), prevCell1.capacity*network.timestep/3600.0)
        sendingFlow2= min( sum(prevCell2.vehiclesByGroup.values()), prevCell2.capacity*network.timestep/3600.0)
        
        backWaveRatio = cell.backWaveSpeed/cell.freeFlowSpeed
        receivingFlow = min(cell.capacity*network.timestep/3600.0,
                                backWaveRatio*
                              (cell.maxNoOfVeh-sum(cell.vehiclesByGroup.values())))
        actualFlow1 = 0.0
        actualFlow2 = 0.0
        if sendingFlow1+sendingFlow2<receivingFlow:
            actualFlow1=sendingFlow1; actualFlow2=sendingFlow2;
        else:
            #receiving flow is limiting so only allow flows in proportion of capacities
            propCell1 = prevCell1.capacity/(prevCell1.capacity+prevCell2.capacity)
            if propCell1*receivingFlow <= sendingFlow1 and (1-propCell1)*receivingFlow <= sendingFlow2:
                actualFlow1=propCell1*receivingFlow
                actualFlow2=(1-propCell1)*receivingFlow
            elif propCell1*receivingFlow > sendingFlow1:
                actualFlow1=sendingFlow1
                actualFlow2=receivingFlow-actualFlow1
            else:
                actualFlow2=sendingFlow2
                actualFlow1=receivingFlow-actualFlow2
        
#        if actualFlow1<0.0 or actualFlow2<0.0:
#            print("sendingFlow1 %f, sendingFlow2 %f, receivingFlow%f, actualFlow1%f, actualFlow2 %f" %
#                  (sendingFlow1, sendingFlow2, receivingFlow, actualFlow1, actualFlow2))
#        assert actualFlow1>=0.0,"Negative flow. Fix"
#        assert actualFlow2>=0.0,"Negative flow. Fix"
        
        for groupID in network.groupsByID:
            if sum(prevCell1.vehiclesByGroup.values())>1E-8:
                prevConn1.vehiclesByGroup[groupID]= actualFlow1* prevCell1.vehiclesByGroup[groupID]/sum(prevCell1.vehiclesByGroup.values())
            else:
                prevConn1.vehiclesByGroup[groupID]=0.0
            if sum(prevCell2.vehiclesByGroup.values())>1E-8:
                prevConn2.vehiclesByGroup[groupID]= actualFlow2* prevCell2.vehiclesByGroup[groupID]/sum(prevCell2.vehiclesByGroup.values())
            else:
                prevConn2.vehiclesByGroup[groupID]=0.0
    
    #============Sanity check=========================#
    #needed to correct connector flows which are >-1E-8 and negative, to zero
    
    for conn in network.allCellConnectors:
        for groupID in network.groupsByID:
            if conn.vehiclesByGroup[groupID]<0.0 and abs(conn.vehiclesByGroup[groupID])<1E-8:
                conn.vehiclesByGroup[groupID]=0.0
    
    #=================================================#
    #==update no of vehicles in each cell by group====#
    #=================================================#
    sumVehGP = 0.0
    sumVehHOT = 0.0 #includes on-ramp and off-ramp
    
    sumDensityGP = 0.0  #ratio of current no of vehicles to jam density no of vehs
    sumDensityHOT = 0.0
    sumMaxVehGp = 0.0
    sumMaxVehHOT = 0.0
    countGP=0
    countHOT=0
    
    sumSqMLViol = 0.0
    countMLViol = 0.0
    
    TSTTtemp = 0.0
    for cellID in network.allCellsByID:
        cell = network.allCellsByID[cellID]
        sumVeh=sum(cell.vehiclesByGroup.values())
        TSTTtemp = TSTTtemp + sumVeh*network.timestep #these many vehicles spent net.timestep time in the net
        if cell.cellClass=="HOT" or cell.cellClass=="ONRAMP" or cell.cellClass=="OFFRAMP":
            sumVehHOT = sumVehHOT+ sumVeh
            sumDensityHOT = sumDensityHOT + sumVeh/cell.maxNoOfVeh
            sumMaxVehHOT = sumMaxVehHOT + cell.maxNoOfVeh
            countHOT = countHOT+1
        elif cell.cellClass=="GP":
            sumVehGP = sumVehGP + sumVeh
            sumDensityGP = sumDensityGP + sumVeh/cell.maxNoOfVeh
            countGP = countGP + 1
            sumMaxVehGp = sumMaxVehGp + cell.maxNoOfVeh
        
        if cell.cellClass=="HOT":
            currSpeed = cell.length/cell.getCurrentTravelTime() #km/hr
            if currSpeed< network.minSpeedHOT:
                sumSqMLViol = sumSqMLViol+pow(network.minSpeedHOT-currSpeed, 2)
                countMLViol = countMLViol+1
                #print("--Violation. current speed=",currSpeed," while desired speed=",network.minSpeedHOT)
        for groupID in network.groupsByID:
            incomingFlow =0.0
            outgoingFlow=0.0
            if cell.type=="source":
                meanDemand = cell.sourceDemand[network.currentTime][groupID]
                sampledDemand = np.random.normal(meanDemand, network.stddev_demand)
                if sampledDemand<0.0:
                    sampledDemand=0.0
                incomingFlow = sampledDemand*network.timestep/3600.0
            else:
                for prevConn in cell.prevConnectors:
                    incomingFlow = incomingFlow + prevConn.vehiclesByGroup[groupID]
            for nextConn in cell.nextConnectors:
                outgoingFlow = outgoingFlow + nextConn.vehiclesByGroup[groupID]
            if cell.type=="sink":
                outgoingFlow= cell.vehiclesByGroup[groupID] #all vehicles must exit in the sink cell
                network.vehiclesExited = network.vehiclesExited + outgoingFlow
#                if cell.getId()==3:
#                    print("Incoming flow %f and outgoing flow %f" % (incomingFlow, outgoingFlow))
            if cell.vehiclesByGroup[groupID] + incomingFlow - outgoingFlow < -1*1e-8:
                print("outgoing flow %f, incoming flow %f, and current vehs %f" % (outgoingFlow, incomingFlow, cell.vehiclesByGroup[groupID]))
                sys.exit("Flow is negative in the cell")
            cell.vehiclesByGroup[groupID] = cell.vehiclesByGroup[groupID] + incomingFlow - outgoingFlow
    network.TSTTmeasure = network.TSTTmeasure + TSTTtemp
    network.jah_Static = max(network.jah_Static, (sumVehGP-sumVehHOT))
    #network.jah_Statistic2 = network.jah_Statistic2 + (sumDensityGP/countGP-sumDensityHOT/countHOT)
    network.jah_Statistic2 = network.jah_Statistic2 + (sumVehGP-sumVehHOT)
    
    network.tdjah_nstat.append(sumVehGP/sumMaxVehGp - sumVehHOT/sumMaxVehHOT)
    
    network.sumSqMLViolation = network.sumSqMLViolation + sumSqMLViol
    network.countMLViolation = network.countMLViolation + countMLViol
    
    if network.collectTSdiagramInfo:
        network.collectTimeSpaceDiagramInfo()

if __name__=='__main__':
    #net = Network('DESE')
    net = Network('LBJ')
    temp=net.__dict__ #used for debugging as it lets us see the content of class instance easily

    for i in range(120):
        print("\n=======Timestep %d==========\n" % (i*6))
        propogateFlow(net)
        net.currentTime = net.currentTime + net.timestep
        print("Revenue=",net.revenue)
#        if i%20==0:
#            print("\n=======Timestep %d==========\n" % (i*6))
#            net.printFlows()
    net.printTimeSpaceDiagram()