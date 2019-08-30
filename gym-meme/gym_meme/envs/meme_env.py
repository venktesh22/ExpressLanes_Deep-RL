import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np

import os
full_path = os.path.realpath(__file__)
import sys

currentFileDirectory = os.path.dirname(full_path)
sys.path.insert(0,currentFileDirectory)
sys.path.insert(0,currentFileDirectory+"/components/")

from network import Network
import propagateFlow as pf
from math import sqrt

class MemeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    #name= folder name where the inputs are stored
    #objective= type of objective: RMax, TSTTMin, ThptMax, or 
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            print(key,"=",value)
        netName = kwargs['netname']
        seed=kwargs['seed']
        
        
        self.objective = kwargs['objective']
        
        #=====custom code to run multiple runs of jah Thresh=====#
        #if self.objective.lower()=='other':
        self.restrainJAH = False
        self.subtractFromReward= kwargs['jahThresh']
        self.JAHThreshold = 700
        
        self.state = Network(netName,seed)
        #self.state = Network('Mopac')
        self.counter = 0
        self.done = 0
        self.reward = 0
        #self.tollUpdateStep= self.state.tollUpdateStep #every 1 minute toll changes
        
        #other stats
        self.TSTT = 0
        self.totalRev = 0
        self.throughput = 0
        self.jah_stat = -1000 #jam-and-harvest stat; stores ratio of no of vehs in ML to GPL
        self.jah_stat2 = 0 #sum of (differnece of avg density in GPL-ML) for all timesteps
        self.remainingVeh = 0
        self.sumSqMLVio = 0
        self.countMLVio = 0
        #self.render()
#        self.restrainJAH = False
        
#        self.previousToll=[]


    def step(self, target):
        if self.done == 1:
            #print("End of timehorizon reached")
            return [np.array(self.getObservations()), self.reward, bool(self.done), "Extra info"]
        else:
            assert len(target)==len(self.state.tollCells),"Dimension mismatch for tolls"
            target=np.array(target)
            
            #scale between minimum and maximum
            target[target< self.state.tollMin]=self.state.tollMin
            target[target> self.state.tollMax]=self.state.tollMax
#            print("===The toll is ====",target)
            
#            if self.state.currentTime==0:
#                target[target< self.state.tollMin]=self.state.tollMin
#                target[target> self.state.tollMax]=self.state.tollMax
#            else:
#                for i in range(len(target)):
#                    if target[i]> self.previousToll[i]+0.25:
#                        target[i]=max(min(self.previousToll[i]+0.25, self.state.tollMax), self.state.tollMin)
#                    elif target[i]< self.previousToll[i]-0.25:
#                        target[i]=max(min(self.previousToll[i]-0.25, self.state.tollMax), self.state.tollMin)
#                    else:
#                        target[i]=max(min(target[i], self.state.tollMax), self.state.tollMin)
            
            self.state.setToll(target)
#            self.previousToll = target.copy()
            
            for i in range(0,self.state.tollUpdateStep,self.state.timestep):
                #TSTT calculations
                noVehsInSystem = 0
                for cell in self.state.allCellsByID:
                    noVehsInSystem = noVehsInSystem + sum(self.state.allCellsByID[cell].vehiclesByGroup.values())
                #self.TSTT = self.TSTT + noVehsInSystem* self.state.timestep #each veh stays in system for 6 seconds
                #print("propagating flow...")
                pf.propogateFlow(self.state)
                self.state.currentTime = self.state.currentTime + self.state.timestep
                
                #throughput update
            
            #=======OBJECTIVE===========#
            if self.objective.lower()=="revmax":
                self.reward = self.state.revenue
            elif self.objective.lower()=="tsttmin":
                self.reward = 100-1*self.state.TSTTmeasure/3600.0 #use hour units
                #self.reward = 360000-1*self.state.TSTTmeasure #use sec units
            elif self.objective.lower()=="thptmax":
                self.reward = self.state.vehiclesExited
            else:
                #if the objective is any other custom value
                #========TSTT MIN OBJECTIVE WAY 2======#
                #compute number of vehs in HOT cell
    #            noVehTollCell=0
    #            for cell in self.state.tollCells:
    #                noVehTollCell = noVehTollCell+sum(cell.vehiclesByGroup.values())
    #            self.reward = noVehTollCell
                #========TSTT MIN OBJECTIVE WAY 3======#
                #self.reward = noVehTollCell
                #========Rev max + min TSTT as objective=====#
#                weight=0.4 #weightage on revenue part
#                self.reward = weight*self.state.revenue + (1-weight)*(100-1*self.state.TSTTmeasure/3600.0)
                lambdaVal=0.175
                self.reward = 50+(lambdaVal*self.state.revenue + (-1)*(self.state.TSTTmeasure/3600.0))
                #======Rev max+JAH correction======#
                #print("Rev=",self.state.revenue," and jah_stat/10k=",self.state.jah_Statistic2/10000)
                #self.reward = self.state.revenue - self.state.jah_Statistic2/50
#            

            
            #total revenue update
            self.totalRev = self.totalRev + self.state.revenue #no discounting
            self.TSTT = self.TSTT + self.state.TSTTmeasure
            self.throughput = self.throughput + self.state.vehiclesExited
            self.jah_stat = max(self.jah_stat, self.state.jah_Static)
            self.jah_stat2 = self.jah_stat2 + self.state.jah_Statistic2
            self.sumSqMLVio = self.sumSqMLVio + self.state.sumSqMLViolation
            self.countMLVio = self.countMLVio + self.state.countMLViolation
            
            if self.state.currentTime%60==-1 and self.state.currentTime < -1:
                self.render() #render under this constraint
            self.state.revenue=0.0
            self.state.TSTTmeasure=0.0
            self.state.vehiclesExited=0.0
            self.state.jah_Static=-1000
            self.state.jah_Statistic2=0
            self.state.sumSqMLViolation=0
            self.state.countMLViolation=0

        if self.state.currentTime >= self.state.simTime: #ensure that simulation runs for 30 min
            if self.restrainJAH:
                if self.jah_stat> self.JAHThreshold:
                    self.reward = self.reward - self.subtractFromReward
            self.done=1
            lastVeh = 0
            for cellID in self.state.allCellsByID:
                cell = self.state.allCellsByID[cellID]
                lastVeh = lastVeh+sum(cell.vehiclesByGroup.values())
            self.remainingVeh = lastVeh
        else:
            self.done=0
        return [np.array(self.getObservations()), self.reward, bool(self.done), "Extra Info"]

    def reset(self):
        #print("=====Reseting Network=====")
        self.state.resetNetwork() #network is reset
        self.counter = 0
        self.done = 0
        self.reward = 0
        
        self.TSTT = 0
        self.totalRev = 0
        self.throughput = 0
        self.jah_stat = -1000 #jam-and-harvest stat; stores ratio of no of vehs in ML to GPL
        self.jah_stat2 = 0
        self.sumSqMLVio = 0 
        self.countMLVio = 0
        #I must return an observation after reseting the state
        return np.array(self.getObservations())

    def render(self):
        self.state.printTimeSpaceDiagram()
        
    #returns array of observations which is total number of vehs on each link being observed
    #we will experiment with other observations later
    def getObservations(self):
        observation=[]
        for l in self.state.observationLinksOrdered:
            totalVeh=0
            for cell in l.allCells:
                totalVeh = totalVeh + sum(cell.vehiclesByGroup.values())
            # @todo: make observations stochastic
            observation.append(totalVeh)
        observation.append(self.state.currentTime)
        if np.any(np.isnan(np.array(observation))):
            self.render()
            sys.exit("Nan value detected for observation...terminating!")
        return observation
    
    def getHOTDensityData(self):
        return self.state.allHOTTimestampedDensities
    
    def getGPDensityData(self):
        return self.state.allGPTimestampedDensities
    
    def getAllOtherStats(self):
        #return TSTT, total revenue, throughput, and ratio of no of vehs in ML to GPL (lower value indicative of jam-and-harvest nature)
        #return [self.TSTT, self.totalRev, self.throughput, max(self.jah_stat), min(self.jah_stat), np.mean(np.array(self.jah_stat))]
        RMSE_MLvio=0.0
        if self.countMLVio>1E-3:
            RMSE_MLvio = sqrt(self.sumSqMLVio/self.countMLVio)
        perCentTimeVio = self.countMLVio*100/(len(self.state.HOTCellIDs)*self.state.simTime/self.state.timestep)
        #print("===SumSqMLVio=", self.sumSqMLVio, "==countMLvio=",self.countMLVio,"=====percenttimeVio=",perCentTimeVio,"%======")
        return [self.TSTT/3600.0, self.totalRev, self.throughput, self.jah_stat, self.remainingVeh, self.jah_stat2/1000, RMSE_MLvio, perCentTimeVio, max(self.state.tdjah_nstat)]
    
    @property
    def action_space(self):
        tolls = spaces.Box(low=self.state.tollMin, high=self.state.tollMax, shape=(len(self.state.tollCells),))
        return tolls

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(len(self.state.observationLinksOrdered)+1,))

##==debugging phase===
#if __name__=='__main__':
#    env=MemeEnv()
#    ob, r, d, e = env.step([0.1,0.1])
#    print("observation %s, reward= %f" % (ob, r))