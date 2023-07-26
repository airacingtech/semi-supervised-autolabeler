import sys

from regex import W
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from aot_tracker import get_aot
import numpy as np
from tool.segmentor import Segmentor
from tool.detector import Detector
from tool.transfer_tools import draw_outline, draw_points
import tool.roar_tools as rt
from tqdm import tqdm
from seg_track_anything import draw_mask
from SegTracker import SegTracker
import pickle

class RoarSegTracker(SegTracker):
    def __init__(self,segtracker_args, sam_args, aot_args) -> None:
        # Roar Lab semantic Segmentation added var
        self.label_to_color = {} #associate class with color id
        self.class_obj = {} #associate obj_id with class label or name
        self.key_frame_arr = [] #keep track of frame idx of keyframes
        self.start_frame_idx = 0 #what cvat frame does tracker start at
        self.end_frame_idx = 0 #what cvat frame does tracker end at
        self.key_frame_to_masks: dict[int, dict[int, rt.MaskObject]] = {} #keep track of MaskObjects for each keyframe
        self.curr_key_frame_idx = self.start_frame_idx
        self.img_dim = (1920, 1080)
        super().__init__(segtracker_args, sam_args, aot_args)
    
    def set_label_to_color(self, label_to_color):
        self.label_to_color = label_to_color
    
    def get_label_to_color(self):
        return self.label_to_color
    
    def set_curr_key_frame(self, curr_key_frame_idx: int):
        self.curr_key_frame_idx = curr_key_frame_idx
    def get_curr_key_frame(self):
        return self.curr_key_frame_idx
    def set_key_frame_to_masks(self, key_frame_to_masks):
        self.key_frame_to_masks = key_frame_to_masks
        self.key_frame_arr = list(self.key_frame_to_masks.keys())
        
    def get_key_frame_to_masks(self):
        return self.key_frame_to_masks
    
        
    def set_key_frame_arr(self, key_frame_arr):
        self.key_frame_arr = key_frame_arr
    
    def get_key_frame_arr(self):
        return self.key_frame_arr       
    
    def set_img_dim(self, img_dim):
        self.img_dim = img_dim
    
    def get_img_dim(self):
        return self.img_dim
    
    
    def start_seg_tracker_for_cvat(self, annotation_dir=""):
        '''Prime seg tracker for use with cvat export
        Arguments:
            start_idx: int
            end_idx: int
        Return:
            none
        '''
        if annotation_dir == "":
            return
        masks, labels_dict, img_dim, start_frame, stop_frame = rt.xml_to_masks(annotation_dir)
        key_frame_idx_to_mask_objs = rt.masks_to_mask_objects(masks, labels_dict, img_dim)
        self.start_frame_idx = int(start_frame)
        self.end_frame_idx = int(stop_frame)
        self.img_dim = (img_dim['width'], img_dim['height'])
        self.set_label_to_color = labels_dict
        self.set_key_frame_to_masks = key_frame_idx_to_mask_objs
        self.set_key_frame_arr = list(self.set_key_frame_to_masks.keys())
        
    def create_origin_mask(self, key_frame_idx=0) ->  np.array:
        """Create origin mask for tracking
        
        Arguments:
            key_frame_idx: int value of desired key frame with annotations to create origin mask
            
        Return:
            None
        """
        
        mask_objects = self.get_key_frame_to_masks().get(key_frame_idx)
        if mask_objects is None:
            return
        self.origin_merged_mask = np.zeros(self.img_dim, dtype=np.uint8)
        
        for mask_object in tqdm(mask_objects):
            self.origin_merged_mask = self.origin_merged_mask + mask_object.get_mask_array()
            # self.class_obj[mask_object.get_id()] = mask_object.get_label()
        self.first_frame_mask = self.origin_merged_mask
        return self.origin_merged_mask
            
        
        
                                                                       
        
    def add_reference_with_label(self, frame, mask, label, frame_step=0):
        """
        Add objects under same label in mask for tracking
        Arguments:
            frame: numpy array (h,w,3)
            mask: MaskObject
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
            
        self.add_reference(frame, mask.get_mask_array(), frame_step=frame_step)
        
    
    def separate_masks(self, img: np.array) -> list:
        """Separate unique masks from image.
        
        Takes a numpy array and separates it into multiple binary masks.
        
        Parameters:
        img -- Numpy array of dimensions (height, width) with unique values for different masks.
        
        Returns:
        masks -- List of binary masks with the same dimensions as img.
        """
        
        unique_values = np.unique(img)
        # We assume background is represented by 0, so we remove it from unique values
        unique_values = unique_values[unique_values != 0]
        
        masks = [(img == value).astype(int) for value in unique_values]
        return masks
    
    def create_mask_objs_from_pred_mask(self, pred_mask, frame):
        """Create MaskObjects from prediction mask
        Arguments:
            pred_mask: numpy array (h,w)
        Return:
            mask_objs: dict of MaskObjects
        """
        mask_objs = {}
        masks = self.separate_masks(pred_mask)
        for mask in tqdm("Creating mask objects from frame: ", masks):
            mask_obj = rt.MaskObject()
            id, mask_img = rt.get_id_from_array(mask)
            # label = self.class_obj[id]
            mask_objs_at_key_frame = self.key_frame_to_masks[self.curr_key_frame_idx]
            label = mask_objs_at_key_frame[id]
            color = self.label_to_color[label]
            mask_obj.turn_mask_array_to_dict(mask_img, id, label, frame, color)
            mask_objs[mask_obj.get_id()] = mask_obj
        return {frame: mask_objs}
            
            


    @staticmethod
    def store_data(roarsegtracker_obj: RoarSegTracker):
        """Save important data to compressed map object
        Arguments:
            none
        Return:
            dict {str: container}
        """
        return pickle.dumps(roarsegtracker_obj)
    @staticmethod
    def load_data(data):
        """Load important data from compressed map object
        Arguments:
            data (str): string od serialized data for RoarSegTracker object
        Return:
            RoarSegTracker object
        """
        return pickle.loads(data)
    
    def add_mask(self, mask_obj: rt.MaskObject):
        #TODO: implement adding mask to current key frame overall mask?
        mask = mask_obj.get_id() * mask_obj.get_mask_array()
        self.origin_merged_mask = self.origin_merged_mask + mask
    
    def set_key_frame(self, frame_id):
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
        if self.key_frame_to_masks.get(frame_id) is None:
            return
        
        for mask_obj in self.key_frame_to_masks.get(frame_id):
            self.add_mask(mask_obj)
    


            
    