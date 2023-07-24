import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
from tool.segmentor import Segmentor
from tool.detector import Detector
from tool.transfer_tools import draw_outline, draw_points
import tool.roar_tools as rt
import cv2
from seg_track_anything import draw_mask
from SegTracker import SegTracker

class RoarSegTracker(SegTracker):
    def __init__(self,segtracker_args, sam_args, aot_args) -> None:
        # Roar Lab semantic Segmentation added var
        self.label_to_color = {} #associate class with color id
        self.class_obj = {} #associate class name with object
        self.key_frame_arr = [] #keep track of frame idx of keyframes
        self.start_frame_idx = 0
        self.end_frame_idx = 0
        super().__init__()
        
    def start_seg_tracker_for_cvat(self, start_idx, end_idx):
        '''Prime seg tracker for use with cvat export
        Arguments:
            start_idx: int
            end_idx: int
        Return:
            none'''
        self.start_frame_idx = start_idx
        self.end_frame_idx = end_idx
        
    def add_reference_with_label(self, frame, mask, label, frame_step=0):
        """
        Add objects under same label in mask for tracking
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
            label: (str: name, id: color_id)
        """
        #convert hex to rgb if not in (r, g, b) format
        # assert(type(label) is list or isinstance(label, np.ndarray))
        obj_class = label[0]
        obj_color = label[1]
        
        if type(obj_color) is str:
            obj_color = rt.hex_to_rgb(obj_color)
        if self.label_to_color.get(obj_class) is None:
            self.label_to_color[obj_class] = obj_color
            
        if self.class_obj.get(obj_class) is None:
            self.class_obj[obj_class] = obj_color
        self.add_reference(frame, mask, frame_step=frame_step)
    
    def add_mask(self, frame, mask):
        #TODO: implement adding mask to current key frame overall mask?
        return
    
    def set_key_frame(self, frame, masks, labels):
        """Sets origin_merged_mask with all annotated 
        masks for the given key frame. Replaces seg function 
        as annotation is done with CVAT instead of seg everything.

        Args:
            frame np array (h, w, 3): 3 dimensional numpy array of image
            masks list [(h, w), ]: list of masks in same order as labels
            labels list [(str: name, id: color_id), ]: associative class 
                                                    label and color id for masks
        Return:
            origin_merged_mask: numpy array (h,w)      
        """