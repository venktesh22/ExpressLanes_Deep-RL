#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:37:30 2019

@author: vpandey
"""

#import numpy as np
#from node import Node

class Link:
    def __init__(self, i, fromN, toN, length=0.15, cap=2200, jamD=165, ffs=90, bws=30, lC="GP"):
        self.id = i
        self.fromNode= fromN
        self.toNode= toN
        self.length=length
        self.capacity=cap
        self.jamDensity = jamD
        self.FFS=ffs #free-flow speed
        self.BWS=bws #back-wave speed
        self.linkClass=lC #GP, HOT, ONRAMP, OFFRAMP, INLINK, OUTLINK
        self.noOfCells=0
        self.allCells=[]
        self.fromNode.addLink(self)
        self.toNode.addLink(self)

    def __repr__(self):
        return "(Link" + str(self.id) + ")"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.id