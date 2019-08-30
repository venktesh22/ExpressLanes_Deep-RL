#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:32:36 2019

@author: vpandey
"""
import numpy as np
#from link import Link

class Node:
    def __init__(self, i):
        self.id = i
        self.incoming= []
        self.outgoing= []

    def __repr__(self):
        return "(Node" + str(self.id) + ")"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.id

    def addLink(self, l):
        if l.fromNode== self:
            self.outgoing.append(l)
        elif l.toNode==self:
            self.incoming.append(l)