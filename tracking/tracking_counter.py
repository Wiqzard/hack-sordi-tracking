from dataclasses import dataclass



@dataclass
class RackDetection:
    rack_name : str
    rack_conf : float
    N_full_KLT: int
    N_empty_KLT : int
    N_Pholders: int
    shelf_N_Pholders: dict


@dataclass
class RackTracker:
    tracker_id : int
    class_id : int
    rack_conf : float
    shelves = {1 : {"N_full_KLT" : 0, "N_empty_KLT" : 0},
               2 : {"N_full_KLT" : 0, "N_empty_KLT" : 0},
               3 : {"N_full_KLT" : 0, "N_empty_KLT" : 0},
               4 : {"N_full_KLT" : 0, "N_empty_KLT" : 0},
               5 : {"N_full_KLT" : 0, "N_empty_KLT" : 0}} 
