import sys

from regex import W
from sympy import N
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
        self.aot_args = aot_args
        self.class_obj = {} #associate id with og id output by tracker
        self.key_frame_arr = [] #keep track of frame idx of keyframes
        self.start_frame_idx = 0 #what cvat frame does tracker start at
        self.end_frame_idx = 0 #what cvat frame does tracker end at
        self.key_frame_to_masks: dict[int, dict[int, rt.MaskObject]] = {} #keep track of MaskObjects for each keyframe
        self.curr_key_frame_idx = self.start_frame_idx
        self.img_dim = (1920, 1080)
        # self.tracker_no_work = []
        self.blacklist = "HUD"
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
    def get_start_frame_idx(self):
        return self.start_frame_idx
    def get_end_frame_idx(self):
        return self.end_frame_idx
    
    def new_tracker(self):
        self.tracker = get_aot(self.aot_args)
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
        key_frame_idx_to_mask_objs = rt.masks_to_mask_objects(masks, 
                                                              labels_dict, img_dim, int(start_frame), blacklist=self.blacklist)
        self.start_frame_idx = int(start_frame)
        self.end_frame_idx = int(stop_frame)
        self.img_dim = (img_dim['height'], img_dim['width'])
        self.set_label_to_color(labels_dict)
        self.set_key_frame_to_masks(key_frame_idx_to_mask_objs)
        self.set_key_frame_arr(list(self.get_key_frame_to_masks().keys()))
        
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
        origin_merged_mask = np.zeros(self.img_dim, dtype=np.uint8)
        self.reference_objs_list = mask_objects
        self.class_obj={}
        keys = list(mask_objects.keys())
        keys.sort()
        min_val = 0
        for i in range(len(keys)):
            if i == 0:
                min_val = keys[i]
                self.class_obj[keys[i] + 1 - min_val] = keys[i] + 1
            else:
                self.class_obj[keys[i] + 1 - min_val] = keys[i] + 1
        for mask_object in tqdm(mask_objects.values()):
            img = mask_object.get_mask_array() #adds + 1 to orignal id to make sure not 0 like background
            filter_mask = img != 0
            test_img_correct = np.unique(img)
            origin_merged_mask[filter_mask] = img[filter_mask]
            test_img_correct_post = np.unique(origin_merged_mask)
            
            
            # self.class_obj[mask_object.get_id()] = mask_object.get_label()
        self.first_frame_mask = origin_merged_mask
        self.origin_merged_mask = origin_merged_mask
        return origin_merged_mask
            
    #override  
    def get_tracking_objs(self):
        return list(self.reference_objs_list.values())
    #override
    def get_obj_num(self):
        objs = list(self.reference_objs_list.keys())
        return len(objs)
                                                                       
        
    def add_reference_with_label(self, frame, mask, frame_step=0):
        """
        Add objects under same label in mask for tracking
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
            label: (str: name, id: color_id)
        """
        #convert hex to rgb if not in (r, g, b) format
        # assert(type(label) is list or isinstance(label, np.ndarray))
        # obj_class = label[0]
        # obj_color = label[1]
        
        # if type(obj_color) is str:
        #     obj_color = rt.hex_to_rgb(obj_color)
        # if self.label_to_color.get(obj_class) is None:
        #     self.label_to_color[obj_class] = obj_color
        self.curr_idx = self.get_obj_num() + 1 
         
        self.tracker.add_reference_frame(frame, mask, self.get_obj_num(), frame_step=frame_step)
        
        
    
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
        if unique_values[0] == 0:
            unique_values = np.delete(unique_values, 0)
        
        masks = [[(img == value).astype(int), value] for value in unique_values]
        return masks
    
    def create_mask_objs_from_pred_mask(self, pred_mask, frame: int):
        """Create MaskObjects from prediction mask
        Arguments:
            pred_mask: numpy array (h,w)
        Return:
            mask_objs: dict of MaskObjects
        """
        mask_objs = {}
        masks_and_ids = self.separate_masks(pred_mask)
        for mask_and_id in tqdm(masks_and_ids, "Creating mask objects from frame: "):
            # test_mask = np.unique(mask)
            mask_obj = rt.MaskObject()
            mask_img, id = mask_and_id
            id = self.class_obj[id]
            id = mask_obj.find_orig_id(id) #because obj id 0 exists which is bad when background is also 0, id from get_id_from_array offset by 1
            # label = self.class_obj[id]
            mask_objs_at_key_frame = self.key_frame_to_masks[self.curr_key_frame_idx]
            label = mask_objs_at_key_frame[id].get_label()
            color = self.label_to_color[label]['color']
            mask_obj.turn_mask_array_to_dict(mask_img, id, label, frame, color)
            mask_obj.start_0 = mask_objs_at_key_frame[id].start_0
            mask_objs[mask_obj.get_id()] = mask_obj
        return mask_objs
            
            


    @staticmethod
    def store_data(roarsegtracker_obj):
        """Save important data to compressed map object
        Arguments:
            none
        Return:
            byte str: string of serialized data for RoarSegTracker object
        """
        return pickle.dumps(roarsegtracker_obj)
    @staticmethod
    def load_data(data):
        """Load important data from compressed map object
        Arguments:
            data (byte str): string of serialized data for RoarSegTracker object
        Return:
            RoarSegTracker object
        """
        return pickle.loads(data)
    
    def add_mask(self, mask_obj):
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
    


            
    