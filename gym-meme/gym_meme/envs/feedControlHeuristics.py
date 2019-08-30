#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:42:17 2019

@author: vpandey
This function implements the density and ratio feedback control heuristics
"""


import gym
import numpy as np
import pandas as pd
#import seaborn as sns; sns.set()
import gym_meme
#import matplotlib.pyplot as plt
import random as random

def runDensityControlHeuristics(seed, netName = 'Mopac', objective='RevMax'):
    env = gym.make('meme-v0', netname= netName, objective=objective, seed=seed)
    debugModeRenderingOn = False
    
    #create a mapping of tollCell to link it is observing
    tollCellToObservedLink = {}
    for cell in env.state.tollCells:
        #cell= env.state.allCellsByID[cellID]
        cellID = cell.id
        linkObs = None
        if cell.cellClass== "ONRAMP":
            downstreamHOTNode = (int(cellID/100))%100
            linkObs = env.state.allNodesByID[downstreamHOTNode].outgoing[0]
        elif cell.cellClass== "HOT":
            linkObs = env.state.allLinksByID[int(cellID/100)]
        tollCellToObservedLink[cellID]=linkObs
            
    random.seed(seed)
    
    
    #define the parameters for 
    regParam = np.array([0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4]) 
#    regParam = np.array([0.01])
    #regParam = np.array([0.01])
    minToll = env.state.tollMin
    maxToll = env.state.tollMax
    
    import time
    timeSec = round(time.time())
    fileName = "outputs/"+netName+"/DensityHeuristic"+"_"+str(timeSec)+".txt"
    f = open(fileName, "w")
    f.write("scalingFactor \t param \t TSTT(hr) \t Rev($) \t Thpt \t JAH \t RemVeh \t JAH_2 \t RMSE \t PercentVio \t tdJAHNormMax \t TollProfile\n")
    
    desiredVehiclePerCapacity={} #mapping of toll cell to no of desired vehicle on the corresponding link
    for tollCell in env.state.tollCells:
        link = tollCellToObservedLink[tollCell.id]
        desiredVehsML = 0.0
        for cell in link.allCells:
            desiredVehsML = desiredVehsML + cell.capacity*env.state.timestep/3600
        desiredVehiclePerCapacity[tollCell.id]=desiredVehsML
    
    for scalingFactor in np.linspace(0.5,1.0,6):
        for param in regParam:
            for i in np.linspace(minToll,maxToll,5):
                #actionSequence = []
                runningReward = 0
                env.reset()
                actionVec=[]
                for tollCell in env.state.tollCells:
                    actionVec.append(random.random()*(maxToll-minToll)+minToll)
                    #actionVec.append(i)
                    #actionVec.append(minToll)
                
                #action = 0.1
                while True:
                    #initialize toll in first step
                    
                    #actionSequence.append(actionVec.copy())
                    #step the environment with that toll
                    observation_, reward, done, info = env.step(actionVec)
                    
                    runningReward = runningReward+reward
                    noVehsML = 0.0 #sum(observation_[1:2:2]) #collect total no of vehs on HOT cell...specific to HOT
                    vehMLVec=[]
                    desiredVehMLVec=[]
                    
                    index=0
                    for tollCell in env.state.tollCells:
                        link = tollCellToObservedLink[tollCell.id]
                        indexOfObservedLinkInObsVector = env.state.observationLinksOrdered.index(link)
                        noVehsML = observation_[indexOfObservedLinkInObsVector]
                        desiredVehsML = scalingFactor*desiredVehiclePerCapacity[tollCell.id]
                        action = actionVec[index] + param*(noVehsML-desiredVehsML)
                        if action>maxToll:
                            action=maxToll
                        elif action<minToll:
                            action=minToll
                        actionVec[index]=action
                        vehMLVec.append(noVehsML)
                        desiredVehMLVec.append(desiredVehsML)
        #                print('Link ',link,' has current Vehs=',noVehsML,' while the desired no is ', desiredVehsML
        #                      , ' so the toll charged is ', action)
                        index += 1
                    #print("currentVehML=", vehMLVec, " and desiredVehML=", desiredVehMLVec)
                    
                    #record the 
                    if done:
                        break
                aS = env.getAllOtherStats() #all stats
                f.write("%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %s \n" 
                        % (scalingFactor, param, aS[0], aS[1], aS[2], aS[3], aS[4], aS[5], aS[6], aS[7], aS[8], env.state.tollProfile))
                print("scaling Factor=",scalingFactor," reg param=",param," rev=",aS[1], " and TSTT=", aS[0])#, " and tolls=",actionSequence)
                #print("env stats=", aS)
                if debugModeRenderingOn:
                    env.state.plottdJAH()
                    env.render()
    #            break
    #        break
    f.close()

def selectTollsRandomly(seed, numberItr=1000, netName = 'LBJ'):
    env = gym.make('meme-v0', netname=netName, objective='RevMax', seed=seed)
    writeOutputFile = True
    debugModeRenderingOn = False
    
    
    random.seed(seed)
    minToll = env.state.tollMin
    maxToll = env.state.tollMax
    
    maxRevSoFar = -10000
    maxRevProfile = []
    maxRevAllStats = []
    
    minTSTTSoFar = 100000000
    minTSTTProfile = []
    minTSTTAllStats = []
    
    import time
    timeSec = round(time.time())
    fileName = "outputs/"+netName+"/RandomResults"+"_"+str(timeSec)+".txt"
    if writeOutputFile:
        f=open(fileName,"w+") #reopen in write mode
    
    print("ItrNo\tAllStats\tMaxRevSoFar\tMinTSTTSoFar")
    if writeOutputFile:
        f.write("ItrNo \t TSTT(hr) \t Rev($) \t Thpt \t JAH \t RemVeh \t JAH_2 \t RMSE \t PercentVio \t tdNormJAHMax \t MaxRevSoFar\t MinTSTTSoFar\n")
    
    for itrNo in range(numberItr):
        
        actionSequence = []
        #runningReward = 0
        env.reset()
        
        prevAction = []
        for c in range(len(env.state.tollCells)):
            prevAction.append(random.random()*(maxToll-minToll)+minToll)
#            prevAction.append(maxToll)
        
        while True:
            action=[]
            if itrNo%10==0 or itrNo%10==4:
                for c in range(len(env.state.tollCells)):
                    action.append(random.random()*(maxToll-minToll)+minToll) #min toll is $0.1, max toll is $3
            elif itrNo%10==1 or itrNo%10==5 or itrNo%10==8:
                for i in range(len(prevAction)):
                    newMaxToll = min(prevAction[i]+0.25, maxToll)
                    newMinToll = max(prevAction[i]-0.25, minToll)
                    action.append(random.random()*(newMaxToll-newMinToll)+newMinToll)
            elif itrNo%10==2 or itrNo%10==6 or itrNo%10==9:
                for i in range(len(prevAction)):
                    newMaxToll = min(prevAction[i]+0.75, maxToll)
                    newMinToll = max(prevAction[i]-0.75, minToll)
                    action.append(random.random()*(newMaxToll-newMinToll)+newMinToll)
            elif itrNo%10==3 or itrNo%10==7:
                action = prevAction.copy() #simulate constant tolls as well
            actionSequence.append(action)
            #step the environment with that toll
            observation_, reward, done, info = env.step(action)
            prevAction = action.copy()
            #runningReward = runningReward+reward
            #record the 
            if done:
                break
        if debugModeRenderingOn:
            env.state.plottdJAH()
            env.render()
        #[TSTT, rev, throughput, jah_stat] 
        allStatsVector = env.getAllOtherStats()
        if allStatsVector[1]>maxRevSoFar:
            maxRevSoFar=allStatsVector[1]
            maxRevProfile=env.state.tollProfile
            maxRevAllStats= allStatsVector
        if allStatsVector[0]<minTSTTSoFar:
            minTSTTSoFar=allStatsVector[0]
            minTSTTProfile=env.state.tollProfile
            minTSTTAllStats = allStatsVector
        print(itrNo,'\t',allStatsVector,'\t',maxRevSoFar,'\t',minTSTTSoFar, '\t', (round(time.time())-timeSec)) #, '\t', actionSequence, '\t', env.state.tollProfile)
        aS = allStatsVector
#        print("%d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f" 
#                    % (itrNo, aS[0], aS[1], aS[2], aS[3], aS[4], aS[5], aS[6], aS[7], maxRevSoFar, minTSTTSoFar))
        if writeOutputFile:    
            f.write("%d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n" 
                    % (itrNo, aS[0], aS[1], aS[2], aS[3], aS[4], aS[5], aS[6], aS[7], aS[8], maxRevSoFar, minTSTTSoFar))
        #f.write("%d \t %s \t %f \t %f \n" % (itrNo,allStatsVector,maxRevSoFar,minTSTTSoFar))
    print("===Max rev profile====")
    print("Revenue = $",maxRevSoFar)
    print("Toll profile:\n",maxRevProfile)
    print("AllStats (TSTT, rev, throughput, jah_stat)", maxRevAllStats)
    
    
    
    print("===Min TSTT profile====")
    print("TSTT = ",minTSTTSoFar," hrs")
    print("Toll profile:\n",minTSTTProfile)
    print("AllStats (TSTT, rev, throughput, jah_stat): ", minTSTTAllStats)
    
    if writeOutputFile:
        f.write("\n\n===Max rev profile====\n")
        f.write("Revenue = $%f \n" % maxRevSoFar)
        f.write("Toll profile:\n %s \n" % maxRevProfile)
        f.write("AllStats (TSTT, rev, throughput, jah_stat)= %s \n" % maxRevAllStats)
        
        f.write("\n\n===Min TSTT profile====\n")
        f.write("TSTT = %f hrs \n" % minTSTTSoFar)
        f.write("Toll profile:\n %s \n" % minTSTTProfile)
        f.write("AllStats (TSTT, rev, throughput, jah_stat): %s \n" % minTSTTAllStats)
    
        f.close()

def simulateAGivenTollProfile(tollProfile, netName = 'LBJ'):
    env = gym.make('meme-v0', netname=netName, objective='RevMax', seed=seed)
    #@todo code the simulation of a particular toll profile
    index=0
    while True:
        index = index+1
        #step the environment with that toll
        observation_, reward, done, info = env.step(tollProfile[index])
        #runningReward = runningReward+reward
        #record the 
        if done:
            break
    env.state.plottdJAH()
    env.render()
    allStatsVector = env.getAllOtherStats()
    print("All Stats", allStatsVector)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--netName', type=str, default='Mopac')
    parser.add_argument('--heuristic', type=str, default='Density')
    parser.add_argument('--noItr', type=int, default=100)
    args = parser.parse_args()
    
    if args.heuristic.lower()=='density':
        runDensityControlHeuristics(100, args.netName)
    elif args.heuristic.lower()=='random':
        selectTollsRandomly(100, args.noItr, args.netName)
    else:
        print("Incorrect heuristic input... It has to be either 'Density' or 'Random'...Terminating!")