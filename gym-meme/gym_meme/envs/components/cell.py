#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 07:03:39 2019

@author: vpandey
"""

class Cell:
    def __init__(self, i, link, tp, c, l, ffs, bws, cap, mNOV, isTolled, groupsByID):
        self.id = i
        self.link = link
        self.type=tp #diverge, merge, ordinary, source, sink
        self.cellClass = c #INLINK, GP, HOT, ONRAMP, OFFRAMP, OUTLINK (just an extra info)
        self.length=l #km
        self.freeFlowSpeed=ffs #km/hr
        self.backWaveSpeed= bws #km/hr
        self.capacity= cap #veh/hr (not in vehicles/timestep so be cautious)
        self.maxNoOfVeh = mNOV #this is based on deltaX of the network
        self.isTolled = isTolled
        self.prevConnectors = []
        self.nextConnectors = []
        self.vehiclesByGroup = {} #defines no of vehicles by each
        #self.nextVehiclesByClass = {}
        self.defineVehicleByGroups(groupsByID)
        
        ##diverge cell parameters
        self.endCell = None
        self.pathToEndCell =[]
        
        ##toll cell parameter
        self.toll=0.0
        
        ##source cell demand
        self.sourceDemand= {} #dictionary timestep --> {groupID --> demand} units in veh/hr

    def __repr__(self):
        return "(" + str(self.id) + ")"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.id
    
    def printConnections(self):
        print("Previous cells are %s and next cells are %s and VOT %s" 
              % (self.prevConnectors, self.nextConnectors, self.vehiclesByGroup))
    
    def defineVehicleByGroups(self, groupsByID):
        #votVals = [10,15,20,25,30] #$/hr
        for groupID in groupsByID:
            self.vehiclesByGroup[groupID]=0.0
            #self.nextVehiclesByClass[vot]=0.0
    
    #returns travel time in hour units
    def getCurrentTravelTime(self):
        totalCurrentVeh=sum(self.vehiclesByGroup.values())
        traveltime = max(self.length/self.freeFlowSpeed, totalCurrentVeh/self.capacity);
        if self.maxNoOfVeh <= totalCurrentVeh:
            traveltime=10000.0 #high number
        else:
            traveltime=max(traveltime, (totalCurrentVeh*self.length)/(self.backWaveSpeed*(self.maxNoOfVeh-totalCurrentVeh)))
        return traveltime

    def getToll(self):
        return self.toll
        