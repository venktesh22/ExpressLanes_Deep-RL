#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:42:40 2019
Functions for file reading

@author: vpandey
"""
import csv
import pandas as pd
import re

import os
import sys
import os.path as osp


inputsDirectory = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'inputs')

#path = os.getcwd()
#sys.path.insert(0,path+'/gym-meme/gym_meme/envs/components/')


#sys.path.insert(0,'/Users/vpandey/Box Sync/UT Acads/PhD Dissertation/Modeling/MEME_DRL/gym-meme/gym_meme/envs/components/')

from node import Node
from link import Link

def readLinkFile(netname):
    nodesByID={}
    linksByID={}
    minSpeedLimit=10000 #km/hr
    with open(inputsDirectory+'/'+netname+'/Links.txt','r') as f:
#    with open('/Users/vpandey/Box Sync/UT Acads/PhD Dissertation/Modeling/MEME_DRL/gym-meme/gym_meme/envs/inputs/'+netname+'/Links.txt','r') as f:
        next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for fID, tID, l,c,jD,ffs,bws,cl,ini in reader:
            fID=int(fID); tID=int(tID)
            l= float(l); c=float(c); jD=float(jD)
            ffs=float(ffs); bws=float(bws)
            if fID not in nodesByID:
                n= Node(fID)
                nodesByID[fID]=n
            if tID not in nodesByID:
                n=Node(tID)
                nodesByID[tID]=n
            fromNode = nodesByID[fID]
            toNode = nodesByID[tID]
            linkID = fID*100+tID
            l= Link(linkID, fromNode, toNode, l, c, jD, ffs, bws, cl)
            if ffs<minSpeedLimit:
                minSpeedLimit=ffs
            linksByID[linkID]=l
    return nodesByID,linksByID, minSpeedLimit
    #print(nodesByID[6].outgoing)
    #print(linksByID)

def readVOTFile(netname):
    votProp={}
    with open(inputsDirectory+'/'+netname+'/VOT.txt','r') as f:
        next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for vot, prop in reader:
            vot=float(vot)
            prop=float(prop)
            votProp[vot]=prop
    return votProp

def readDemandFile(netname):
    data=pd.read_csv(inputsDirectory+'/'+netname+'/ODDemandProfile.txt', sep='\t')
    return data

def readObservedLinksFile(netname):
    fileLocation = 'inputsDirectory+'/'+netname+'/ObservedLinkIDs.txt'
    linkIDs=[]
    with open(fileLocation,'r') as f:
        next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for lID in reader:
            #print(lID)
            linkIDs.append(int(lID[0]));
    return linkIDs

def readParametersFile(netname):
    with open(inputsDirectory+'/'+netname+'/Parameters.txt','r') as f:
        #next(f) # skip headings
        #reader=csv.reader(f,delimiter='\t')
        string=f.readlines()
        netname=string[0].split('\t')[1]
        simTime=int(string[1].split('\t')[1])
        timeStep=int(string[2].split('\t')[1])
        tollStep=int(string[3].split('\t')[1])
        demandFactor=float(string[4].split('\t')[1])
        minSpeedLimit=int(string[5].split('\t')[1])
        routeType=string[6].split('\t')[1].split('\n')[0]
        laneStoch = string[7].split('\t')[1].split('\n')[0]
        if laneStoch.lower()=="true" or laneStoch.lower()=="T":
            laneStoch = True
        else:
            laneStoch = False
        stddev_demand = float(string[8].split('\t')[1].split('\n')[0])
#        for lines in f:
#            print(int(lines.split('\t')[1]))
        #simTime = reader['Simulation_time(seconds)']
#        for vot, prop in reader:
#            vot=float(vot)
#            prop=float(prop)
#            votProp[vot]=prop
    return netname,simTime,timeStep,tollStep,demandFactor,minSpeedLimit, routeType, laneStoch, stddev_demand

if __name__ == '__main__':
    data = readDemandFile()