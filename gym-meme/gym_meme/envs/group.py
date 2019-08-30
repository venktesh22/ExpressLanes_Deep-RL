#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:27:51 2019
A group unique identifies all vehicles
@author: vpandey
"""

class Group:
    def __init__(self,id,vot,dest):
        self.id=id
        self.vot=vot
        self.dest=dest
    
    def __repr__(self):
        return "(vot:" + str(self.vot) + ",dest:"+ str(self.dest)+")"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.id