
from typing import Dict, List, TYPE_CHECKING
import string

from .datum import Datum, Relation

if TYPE_CHECKING:
    from .assembly import Assembly


class Component:
    
    def __init__(self, name:str):
        
        self.name=name
        self.origin = [0,0,0]
        self.frame = [0,0,1]
        
        self.primary_datum:Datum=None
        self.datums:Dict[str,Datum] = {}
        self.relations:List[Relation]
        
        self.mating=[]
        
        self.parent=None
    
    def set_parent(self, assy:Assembly)->None:
        
        self.parent = assy
    
    def add_datum(self, datum:Datum, id:str="")->None:
        
        if not id:
            id = string.ascii_uppercase[len(self.datums)]
            
        if not self.primary_datum:
            self.primary_datum=datum
        else:
            datum.set_reference(self.primary_datum)
            
        self.datums[id] = datum
        datum.set_parent(self)


    def __getattr__(self, attr:str):
        if attr in self.datums:
            return self.datums[attr]
        
