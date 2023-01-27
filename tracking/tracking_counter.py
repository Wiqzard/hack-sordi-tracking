from dataclasses import dataclass, field
from typing import List, Dict, Any
from constants.bboxes import CONSTANTS
import json


@dataclass
class RackDetection:
    rack_name: str
    rack_conf: float
    N_full_KLT: int
    N_empty_KLT: int
    N_Pholders: int
    shelf_N_Pholders: dict


empty_shelves = {
    1: {"N_full_KLT": 0, "N_empty_KLT": 0},
    2: {"N_full_KLT": 0, "N_empty_KLT": 0},
    3: {"N_full_KLT": 0, "N_empty_KLT": 0},
    4: {"N_full_KLT": 0, "N_empty_KLT": 0},
    5: {"N_full_KLT": 0, "N_empty_KLT": 0},
}


@dataclass()
class RackTracker:
    """A class to store and process the detection of a rack.
     ...
    Attributes
    ----------
    tracker_id : int
        the id of the tracker
    class_id : int
        the class id of the rack (see constants.bboxes.CONSTANTS.CLASS_NAMES_DICT)
    rack_conf : float
        the confidence of the rack detection
    shelves : Dict[int, Dict[str, int]]
        the number of full and empty KLTs per shelve
    """

    #    def __init__(self, tracker_id: int, class_id: int, rack_conf: float):
    #        self.tracker_id = tracker_id
    #        self.class_id = class_id
    #        self.rack_conf = rack_conf
    #        self.shelves = {
    #            1: {"N_full_KLT": 0, "N_empty_KLT": 0},
    #            2: {"N_full_KLT": 0, "N_empty_KLT": 0},
    #            3: {"N_full_KLT": 0, "N_empty_KLT": 0},
    #            4: {"N_full_KLT": 0, "N_empty_KLT": 0},
    #            5: {"N_full_KLT": 0, "N_empty_KLT": 0},
    #        }

    tracker_id: int
    class_id: int
    rack_conf: float
    shelves: Dict[int, Dict[str, int]] = field(
        default_factory=lambda: {
            1: {"N_full_KLT": 0, "N_empty_KLT": 0},
            2: {"N_full_KLT": 0, "N_empty_KLT": 0},
            3: {"N_full_KLT": 0, "N_empty_KLT": 0},
            4: {"N_full_KLT": 0, "N_empty_KLT": 0},
            5: {"N_full_KLT": 0, "N_empty_KLT": 0},
        }
    )

    @property
    def rack_name(self) -> str:
        return CONSTANTS.CLASS_NAMES_DICT[self.class_id]

    @property
    def N_full_KLT(self) -> int:
        return sum(n_box_full["N_full_KLT"] for n_box_full in self.shelves.values())

    @property
    def N_empty_KLT(self) -> int:
        return sum(n_box_empty["N_empty_KLT"] for n_box_empty in self.shelves.values())

    @property
    def N_Pholders(self) -> int:
        total_boxes = sum(
            value[0] * value[1]
            for key, value in CONSTANTS.NUMBER_BOXES_PER_SHELVE[self.rack_name].items()
        )
        return total_boxes - (self.N_full_KLT + self.N_empty_KLT)

    @property
    def shelf_N_Pholders(self) -> Dict[int, int]:
        return {
            f"shelf_{shelf}": self.placeholders_per_shelf(shelf)
            for shelf, value in CONSTANTS.NUMBER_BOXES_PER_SHELVE[
                self.rack_name
            ].items()
            if self.placeholders_per_shelf(shelf) > 0
        }

    def total_boxes_in_shelf(self, shelf: int) -> int:
        """returns the maximum number of boxes in a shelf"""
        total_boxes_in_shelf = (
            CONSTANTS.NUMBER_BOXES_PER_SHELVE[self.rack_name][shelf][0]
            * CONSTANTS.NUMBER_BOXES_PER_SHELVE[self.rack_name][shelf][1]
        )
        return total_boxes_in_shelf

    def placeholders_per_shelf(self, shelf: int) -> int:
        """returns the number of placeholders in a shelf"""
        n_placeholder_in_shelf = self.total_boxes_in_shelf(shelf) - (
            self.shelves[shelf]["N_full_KLT"] + self.shelves[shelf]["N_empty_KLT"]
        )
        return n_placeholder_in_shelf

    def update_shelves(self, shelf: int, class_id: int) -> None:
        """upates the number of full and empty boxes in a shelf"""
        if self.shelves[shelf]["N_full_KLT"] + self.shelves[shelf][
            "N_empty_KLT"
        ] == self.total_boxes_in_shelf(shelf):
            return
        if class_id == 0:
            self.shelves[shelf]["N_empty_KLT"] += 1
        elif class_id == 1:
            self.shelves[shelf]["N_full_KLT"] += 1


def create_submission_dict(
    scanned_racks: List[RackTracker], maP: float, fps: float
) -> Dict[str, Any]:
    submission_dict = {"eval_video": [], "maP": maP, "FPS": fps}
    for scanned_rack in scanned_racks:
        video_entry = {
            "rack_name": str(scanned_rack.rack_name),
            "rack_conf": float(scanned_rack.rack_conf),
            "N_full_KLT": int(scanned_rack.N_full_KLT),
            "N_empty_KLT": int(scanned_rack.N_empty_KLT),
            "N_Pholders": int(scanned_rack.N_Pholders),
            "shelf_N_Pholders": scanned_rack.shelf_N_Pholders,
        }

        submission_dict["eval_video"].append(video_entry)
    return submission_dict


def write_submission(submission_dict: Dict[str, Any], submission_path: str) -> None:
    with open(submission_path, "w+") as f:
        json.dump(submission_dict, f, indent=4)
