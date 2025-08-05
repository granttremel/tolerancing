

from typing import Dict, List, Tuple
from .component import Component
from .datum import Geometry, Relation, Geometry

class Mate:
    
    def __init__(self, component1, component2, displacement, orientation):
        
        self.c1=component1
        self.c2=component2
        self.d=displacement
        self.o=orientation


class Assembly:
    """
    Container class for components that concerns mating and generating datum transfer maps
    """
    def __init__(self):
        
        self.components:Dict[str,Component] = {}
        self.mates:Dict[str,Mate]={}
        
    def mate(self, comp1, comp2, disp, ori):
        
        newmate = Mate(comp1, comp2, disp, ori)
        
        
        
        pass

    def component_is_defined(self, comp:Component):
        
        
        
        pass

    def add_component(self, comp:Component):
        
        self.components[comp.name]=comp


    def add_relation(self):
        pass
    
    def __getattr__(self, attr:str):
        if attr in self.components:
            return self.components[attr]