from dataclasses import dataclass



@dataclass
class RackDetection:
    rack_name : str
    rack_conf : float
    N_full_KLT: int
    N_empty_KLT : int
    N_Pholders: int
    shelf_N_Pholders: dict

    