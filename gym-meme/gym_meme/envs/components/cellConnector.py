#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:42:14 2019

@author: vpandey
"""

class CellConnector:
    def __init__(self, c1, c2, type):
        self.fromCell = c1
        self.toCell = c2
        self.vehiclesByGroup = {} #defines no of vehicles by each
        self.type=type
    
    def __repr__(self):
        return "(" + str(self.fromCell) +"->" + str(self.toCell)+ ")"
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.fromCell == other.fromCell and self.toCell == other.toCell
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.fromCell.id*100000+self.toCell.id