import numpy as np
import xml.etree.ElementTree as ET
# import datumaro
# import ijson
# from collections import defaultdict
import xmltodict
import copy
import sys
import os
from PIL import Image
import re
from zipfile import ZipFile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aot_tracker import _palette
from scipy.ndimage import binary_dilation
import cv2
import base64
# from RoarSegTracker import RoarSegTracker
# from RoarSegTracker import RoarSegTracker

class MaskObject():
    """MaskObject class representation
    """
    def __init__(self, frame: int = 0, mask: dict = {}, id: int = 0, label="", color_id="", data: dict={}, img_dim = (0, 0)):
            self.frame = frame
            self.mask = mask
            self.id = id
            self.label = label
            self.color = color_id
            self.data = data
            self.img_dim = img_dim
            self.start_0 = False #sometimes converting img back to xml parameters doesnt work if rle starts with 0
    def set_frame(self, frame):
        self.frame = frame
    def get_frame(self) -> int:
        return self.frame
    def set_mask(self, mask: dict):
        self.mask = mask
    def get_mask(self) -> dict:
            return self.mask
    def set_id(self, id: int):
        self.id = id
    def get_id(self) -> int:
        return self.id
    def get_id_for_mask_array(self):
        return self.id + 1 #because id starts at 0 bad for telling from background
    def find_orig_id(self, mask_id):
        return mask_id - 1 #because id starts at 0 bad for telling from background
    def set_label(self, label: str):
        self.label = label
    def get_label(self) -> str:
            return self.label
    def set_color(self, color: str):
        self.color = color
    def get_color(self):
        return self.color
    def set_data(self, data: dict):
        self.data = data
    def get_data(self):
        return self.data
    
    def fix_rle(self, rle):
        if self.start_0:
            rle = np.insert(rle, 0, 0)
        return rle
    
    def copy(self):
        copy_mask_obj = MaskObject(int(self.frame), dict(self.mask), int(self.id), 
                                                  str(self.label), str(self.color), dict(self.data), list(self.img_dim))
        return copy_mask_obj
            
    def get_mask_array(self) -> np.array:
        """Get mask as numpy array.
        
        Returns:
        mask_array -- Numpy array of dimensions (height, width) with mask as 1 * unique id and background as 0.
        """
        mask_array, self.start_0 = mask_to_img(self.data['rle'], self.data['width'], 
                                                  self.data['height'], self.data['top'], 
                                                  self.data['left'], self.img_dim)
        
        max_val = np.max(mask_array)
        mask_array *= self.get_id_for_mask_array()
        max_val_id = np.max(mask_array)
        return mask_array
    def turn_mask_array_to_dict(self, mask_array, id, label, frame, color):
        """Set mask_array to dictionary called self.mask
        
        Arguments:
            mask_array: np.array of dimensions (height, width) with mask as 1 * unique id and background as 0
            id (int): unique id of mask
            label (str): label of mask
            frame (int): frame number of mask
        Returns:
            None
        """
        self.data['rle'], self.data['width'], self.data['height'], \
        self.data['top'], self.data['left'], self.img_dim = img_to_mask(mask_array)
        # self.data['rle'] = self.fix_rle(self.data['rle'])
        self.data['id'] = int(id)
        self.data['label'] = str(label)
        self.data['frame'] = int(frame)
        self.label = str(label)
        self.frame = int(frame)
        self.id = int(id)
        self.color = str(color)
        self.mask = self.data['rle']
        self.set_color(color)
    def __str__(self) -> str:
        return str(self.data)
        
        
def get_image(photo_dir="", frame_num=0) -> np.array:
    """Takes a photo directory and returns a numpy array representing the image.
    
    Arguments:
        photo_dir (str): path to the photo directory. Example: '/home/user/$USERNAME/Downloads/
        frame_num (int): the frame number to specify the image to search for.
        
    Returns:
        np_frame (np.array): numpy array representing the image with dimensions (h, w, 3)."""
    pattern = r'(frame_0*' + str(frame_num) + r'(\.jpg|\.jpeg|\.png|\.PNG|\.bmp))'
    for filename in os.listdir(photo_dir):
        if re.match(pattern, filename):
            path_to_file = os.path.join(photo_dir, filename)
            im_frame = Image.open(path_to_file)
            np_frame = np.array(im_frame)
            return np_frame
def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
# def colorize_mask(pred_mask, color=_palette):
#     save_mask = Image.fromarray(pred_mask.astype(np.uint8))
#     save_mask = save_mask.convert(mode='P')
#     save_mask.putpalette(color)
#     save_mask = save_mask.convert(mode='RGB')
#     return np.array(save_mask)
from PIL import Image
import numpy as np

def colorize_mask(pred_mask, color=(255, 0, 0)):
    """
    Colorizes a mask using a single RGB color.
    
    Args:
    - pred_mask: 2D numpy array representing the mask.
    - color: RGB color to be used for colorizing the mask.
    
    Returns:
    - A colorized version of the mask as a numpy array.
    """
    
    # Create an empty RGB image with the same dimensions as the mask.
    rgb_image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Set the color for the mask pixels.
    rgb_image[pred_mask > 0] = color
        
    return rgb_image

def draw_mask(img, mask, alpha=0.5, id_countour=False, label_to_color: dict = {}):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def make_img_with_masks(id_to_mask_objs: dict = {}, 
                        img: np.array = np.eye(1), id_countour = False, alpha=0.5):
    """Create a RGB Image array with mask objects displayed; with or without contour

    Args:
        id_to_mask_objs (dict, optional): Dictionary of mask object id to mask object. Defaults to {}.
        img (np.array, optional): 3 dimensional matrix for RGB values. Defaults to np.eye(1).
        id_countour (bool, optional): Option to give each mask their designated color of no contour. Defaults to False.
        alpha (float, optional): hyperparameter for alpha channel of image. Defaults to 0.5.

    Returns:
        img_mask: 3 dimensional matrix for RGB values with mask overlay
    """
    img_mask = np.zeros_like(img)
    img_mask = np.copy(img)
    for id, mask_obj in id_to_mask_objs.items():
        color = hex_to_rgb(mask_obj.get_color())
        foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
        mask_array = mask_obj.get_mask_array()
        if id_countour:
            
            #TODO: make binary mask filter to combine img mask with foreground mask?
            binary_mask = (mask_array == id)
            img_mask[binary_mask] = foreground[binary_mask]
            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
        else:
            binary_mask = (mask_array != 0)
            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            foreground = img*(1-alpha)+colorize_mask(mask_array, color=color)*alpha
            img_mask[binary_mask] = foreground[binary_mask]
            img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def hex_to_rgb(hex_color_str):
    """Takes a hexadecimal color string and converts it to RGB.

    Args:
        hex_color_str (str): hexadecimal color string. Example: '#FFFFFF' or 'FFFFFF'

    Raises:
        ValueError: Input string must be a 6 character hexadecimal color string. Example: '#FFFFFF'
    Returns:
        list [int]: returns a list of 3 integers representing RGB color values. Example: [255, 64, 21]
    """
    if hex_color_str[0] == '#':
        hex_color_str = hex_color_str[1:]
    
    if len(hex_color_str) != 6:
        raise ValueError("Input string must be a 6 character hexadecimal color string. Example: '#FFFFFF'")
    
    return [int(hex_color_str[i:i+2], 16) for i in range(0, 6, 2)]


def mask_to_img(rle: np.array, width: int, height: int, top: int, left: int, img_dim: tuple[int, int]) -> np.array:
    """Generate mask image from run length encoding.
    
    Takes a single mask from masks list of xml_to_masks() and converts it to a binary numpy array.
    
    Parameters:
    rle -- Run length encoding of mask.
    width -- Width of mask
    height -- Height of mask
    top -- Top coordinate of mask
    left -- Left coordinate of mask
    img_dim -- Dimensions of image (height, width)
    
    Returns:
    img -- Numpy array of dimensions (img_dim) with mask as 1 and background as 0.
    boolean -- True if start of rle is 0, False otherwise
    """
    
    #blank image
    img = np.zeros(img_dim).astype(int)
    #Decode run length encoding
    bitmap = np.concatenate([np.zeros(n) if i % 2 == 0 else np.ones(n) \
                    for i, n in enumerate(rle)]).reshape(height, width).astype(int)
    #Fill image
    img[top:top+height, left:left+width] = bitmap
    bit_in_img = img[top:top+height, left:left+width]
    assert np.array_equal(bit_in_img, bitmap)
    return img.astype(int), rle[0] == 0
def get_id_from_array(img: np.array):
    
    unique_values = np.unique(img)
    special_array = unique_values[unique_values != 0]
    special_number = special_array[0]
    arr_binary = (img == special_number)
    arr_binary = arr_binary.astype(int)
    return special_number, arr_binary
    
def img_to_mask(img: np.array) -> tuple:
    """Generate run length encoding, mask position and dimensions from binary image.
    
    Takes a binary numpy array and converts it into run length encoding, also finds the position 
    and dimensions of the mask.
    
    Parameters:
    img -- Numpy array of dimensions (height, width) with mask as 1 and background as 0.
    
    Returns:
    rle -- Run length encoding of mask.
    width -- Width of mask
    height -- Height of mask
    top -- Top coordinate of mask
    left -- Left coordinate of mask
    img_dim -- Dimensions of image (height, width)
    """
    if img.ndim != 2:
        raise ValueError('only 2D image supported')
    
    # Find where mask coordinates
    width_nonzero = img.sum(axis=0)
    width_nonzero = width_nonzero.nonzero()
    width_nonzero = width_nonzero[0]
    height_nonzero = img.sum(axis=1).nonzero()[0]
    
    if width_nonzero.size == 0 or height_nonzero.size == 0:
        raise ValueError('The input image does not contain any non-zero elements')

    
    # Find bounding box containing mask
    left = width_nonzero.min()
    top = height_nonzero.min()
    width = width_nonzero.max() - left + 1
    height = height_nonzero.max() - top + 1
    
    # Get mask from image
    mask_box = img[top:top+height, left:left+width].reshape(-1)
    mask_num_pixels = mask_box.shape[0]
    
    # Find run start indexes
    intersections_ixs = np.not_equal(mask_box[:-1], mask_box[1:]).nonzero()[0] + 1
    start_ixs = np.concatenate((np.array([0]), intersections_ixs, np.array([mask_num_pixels])))
    
    # Get Run Lengths
    rle = np.diff(start_ixs)
    
    # Check if mask starts with 1 b/c RLE starts with 0
    if mask_box[0] == 1:
        rle = np.concatenate(([0], rle))
    
    return (rle, width, height, top, left, img.shape)

# About 25x slower than img_to_mask() above
# def img_to_mask(img: np.array) -> tuple:
#     """Generate run length encoding, mask position and dimensions from binary image.
    
#     Takes a binary numpy array and converts it into run length encoding, also finds the position 
#     and dimensions of the mask.
    
#     Parameters:
#     img -- Numpy array of dimensions (height, width) with mask as 1 and background as 0.
    
#     Returns:
#     rle -- Run length encoding of mask.
#     width -- Width of mask
#     height -- Height of mask
#     top -- Top coordinate of mask
#     left -- Left coordinate of mask
#     img_dim -- Dimensions of image (height, width)
#     """
    
#     # Find the bounding box of the mask
#     rows = np.any(img, axis=1)
#     cols = np.any(img, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
    
#     # Extract the mask and its dimensions
#     mask = img[rmin:rmax+1, cmin:cmax+1]
#     height, width = mask.shape
    
#     # Create the run length encoding
#     rle = []
#     current_value = mask[0,0]
#     current_run = 1
#     for value in mask.flatten()[1:]:
#         if value == current_value:
#             current_run += 1
#         else:
#             rle.append(current_run)
#             current_run = 1
#             current_value = value
#     rle.append(current_run)
    
#     return (np.array(rle), width, height, rmin, cmin, img.shape)

def xml_to_masks(filename: str):
    """Parse Annotations.xml file from CVAT for mask recreation.
    
    Parameters:
    filename -- Name of Annotations.xml file
    
    Returns:
    masks -- List of dicts with mask frame, run length encoding, left, top, width, height
        id -- Track ID (0-indexed) corresponds with Object ID in CVAT browser uses 1-indexed
        label -- label of mask
        frame -- Frame number of mask (Corresponds with Image ID)
        rle -- Run length encoding of mask
        left -- Left coordinate of mask
        top -- Top coordinate of mask
        width -- Width of mask
        height -- Height of mask
    labels_dict -- Dictionary with label names as keys
        key -- label name
        value -- {color: Color of mask, id: ID of label}
    img_dim -- Dictionary with keys 'width', 'height'
    start_frame -- Start frame of video
    stop_frame -- Stop frame of video
    """
    # Find Root
    tree = ET.parse(filename)
    root = tree.getroot()
    
    start_frame = root.find('.//start_frame').text
    stop_frame = root.find('.//stop_frame').text
    
    # Find label information
    labels_dict = {}
    for l_id, label in enumerate(root.findall('.//label')):
        l_name = label.find('name').text
        l_color = label.find('color').text
        labels_dict[l_name] = {'color': l_color, 'id': l_id}
    
    # Get Image dimensions
    img_dim_root = root.find('.//original_size')
    img_dim = {'width': int(img_dim_root.find('width').text), 'height':int(img_dim_root.find('height').text)}
    
    track_keys = ['id', 'label']
    mask_keys = ['frame', 'rle', 'left', 'top', 'width', 'height']
    
    masks = []
    # Find keys and cast to correct type
    for track in root.findall('.//track'):
        track_values = [int(track.get(k)) if track.get(k).isdigit() \
            else track.get(k) for k in track_keys]
        mask_values = [int(track.find('mask').get(k)) if track.find('mask').get(k).isdigit() \
            else np.array(track.find('mask').get(k).split(', ')).astype(int) for k in mask_keys]
        masks.append(dict(zip(track_keys + mask_keys,track_values + mask_values)))
    
    return masks, labels_dict, img_dim, start_frame, stop_frame

def masks_to_mask_objects(masks: list, labels_dict: dict, img_dim: dict, start_frame: int, blacklist: str = 'HUD') -> dict[int, dict[int, MaskObject]]:
    """Gathers mask objects from a list of masks and return list of mask objects

    Args:
        masks (list): List of mask dictionaries
        labels_dict (dict): dictionary of labels to color and mask id
        img_dim (dict): dictionary containing dimensions of image 'height' and 'width'

    Returns:
        dict[int, MaskObject]: Dictionary pairing key frame to dict of mask objects mapped by object id
        for that given frame
    """
    # TODO: Add frame range from start_frame to stop_frame
    frame_to_mask_objects = {}
    for mask in masks:
        mask_obj = MaskObject(mask['frame'], mask['rle'], mask['id'], mask['label'], 
                              labels_dict[mask['label']]['color'], mask, (img_dim['height'], img_dim['width']))
        if frame_to_mask_objects.get(mask_obj.frame) is None:
            if mask_obj.get_frame() != start_frame and \
            (mask_obj.get_label() == blacklist):
                continue;
            frame_to_mask_objects[mask_obj.frame] = {mask_obj.get_id(): mask_obj}
        else:
            mask_dict = frame_to_mask_objects[mask_obj.frame]
            mask_dict[mask_obj.get_id()] = mask_obj
            
    return frame_to_mask_objects
import json

def masks_to_xml_with_key_frame():
    return
def masks_to_xml(frame_masks: dict[int, dict[int, MaskObject]], start_frame: int, stop_frame: int, output_filename: str) -> str:
    """Takes a dictionary of frame to mask objects and writes Annotations XML file.

    Args:
        frame_masks (dict): output of masks_to_mask_objects(). Key is frame number, value is dict of mask objects mapped by object id
        output_filename (str): file name to write XML file
        
    Returns:
        str of ile path to teh zip compressed file
    """
    root = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(root, 'template_annotations.xml')
    tree = ET.parse(template_path)
    root = tree.getroot()
    
    # Calculate frame data
    frame_start = start_frame
    frame_end = stop_frame
    frame_count = frame_end - frame_start + 1

    # Update frame data
    root.find('.//size').text = str(frame_count)
    root.find('.//start_frame').text = str(frame_start)
    root.find('.//stop_frame').text = str(frame_end)
    
    label_tuples = []
    img_dim = (0,0)

    # Loop through all frame and all Mask Objects in each frame
    for id_to_mask_objs in frame_masks.values():
        for mask in id_to_mask_objs.values():
            # Store label values
            label_tuple = (mask.label, mask.color)
            if label_tuple not in label_tuples:
                label_tuples.append(label_tuple)
            
            # Store image Dimensions
            if img_dim == (0,0):
                img_dim = mask.img_dim

            # Copy mask data and convert values to strings
            data = copy.deepcopy(mask.data)
            data['rle'] = ', '.join(data['rle'].astype(str))
            out_data = {k: str(v) for k,v in data.items()}
            del data

            # Create Track Element 
            track_dict = {k: out_data.pop(k) for k in ['id', 'label']}
            track_dict['source'] = 'semi-auto'
            track_ele = ET.Element('track', track_dict)

            # Add attributes for first mask
            out_data['keyframe'] = '1'
            out_data['outside'] = '0'
            out_data['occluded'] = '0'
            out_data['z_order'] = '0'
            mask_ele_1 = ET.SubElement(track_ele, 'mask', out_data)

            # Update attributes for second mask
            out_data['outside'] = '1'
            out_data['frame'] = str(int(out_data['frame']) + 1)
            #kevin try edit
            # out_data['frame'] = str(mask.get_frame())
            mask_ele_2 = ET.SubElement(track_ele, 'mask', out_data)

            # Append Track Element to annotations
            root.append(track_ele)
            
    # Update Image Size in XML
    img_size = root.find('.//original_size')
    img_size.find('./height').text = str(img_dim[0])
    img_size.find('./width').text = str(img_dim[1])
    
    # Add label tags to XMl
    labels_tag = root.find('.//labels')
    for label, color in label_tuples:
        label_ele = ET.Element('label')
        label_dict = {'name': label, 'color': color, 'type': 'any', 'attributes':''}

        # Create Element/Text pairs for items in label_dict
        ele_list = [ET.Element(k) for k in label_dict.keys()]
        for i, v in enumerate(label_dict.values()):
            ele_list[i].text = v

        # Add Element/Text pairs to label Element
        for ele in ele_list:
            label_ele.append(ele)

        # Add Element to labels Element
        labels_tag.append(label_ele)
        
    # Pretty Printing
    ET.indent(tree)
    tree.write(output_filename, encoding='utf-8', xml_declaration=True, short_empty_elements=False)
    zip_file_path = os.path.join(os.path.dirname(output_filename), "annotation.zip")
    with ZipFile(zip_file_path, 'w') as zip:
        zip.write(output_filename, arcname=os.path.basename(output_filename))
    return zip_file_path
def get_correct_input(bool_func=lambda x: x, process_function= lambda x: x, question: str = "Question details"):
    repeat = True
    while repeat:
        answer = input(question)
        answer = process_function(answer)
        check = input("is your input: {} correct? (y/n) ".format(answer))
        repeat = ((check == 'y' or check == 'Y') and not bool_func(answer))
    return answer   
def numpy_to_base64(img_np):
    is_success, im_buf_arr = cv2.imencode(".jpg", img_np)
    byte_im = im_buf_arr.tobytes()
    return base64.b64encode(byte_im).decode('utf-8')
if __name__ == '__main__':
    ##TESTING
    masks, labels_dict, img_dim, start_frame, stop_frame = xml_to_masks("/home/roar-nexus/Downloads/annotations.xml")

    # # TODO: Update method after adding frame range
    frame_to_mask_objects  = masks_to_mask_objects(masks, labels_dict, img_dim, 0, '')
    masks_to_xml(frame_to_mask_objects, 0, 3092, '/home/roar-nexus/Segment-and-Track-Anything/roar_annotations/23/output/annotations_output/test_annotations.xml')
    # for key_frame, mask_objects in frame_to_mask_objects.items():

    #     for mask_id, mask_obj in mask_objects.items():
    #         data = mask_obj.get_data()
    #         img = mask_obj.get_mask_array()
    #         rle, width, height, top, left, img_shape = img_to_mask(img)
    #         rle = mask_obj.fix_rle(rle)
    #         assert np.array_equal(rle, data['rle'])
    #         assert width == data['width']
    #         assert height == data['height']
    #         assert top == data['top']
    #         assert left == data['left']
    #         assert np.array_equal(img_shape, mask_obj.img_dim)
        
# masks_to_xml(frame_to_mask_objects, int(start_frame), int(stop_frame), '/home/roar-nexus/Downloads/test_annotations.xml')
# # roarseg = RoarSegTracker(masks, labels, img_dim)
# import xmltodict
# import json
def xml_to_dict(filename: str) -> dict:

    # Parse the XML file into a Python dictionary
    with open("/home/roar-nexus/Downloads/reupload_cvat_test/annotations.xml", 'r') as file:
        xml_string = file.read()
    xml_dict = xmltodict.parse(xml_string)
    return xml_dict

# # Convert the dictionary to a JSON string for pretty printing
# json_str = json.dumps(xml_dict, indent=4)
# print(json_str)

# # Convert the dictionary back to an XML string
def dict_to_xml(xml_dict: dict) -> str:
    xml_string_reconstructed = xmltodict.unparse(xml_dict, pretty=True)
    return xml_string_reconstructed

def write_xml(xml_dict: dict, output_filename: str) -> None:
    xml_string_reconstructed = dict_to_xml(xml_dict)

    # Save the reconstructed XML to a file
    with open('/home/roar-nexus/Downloads/reupload_cvat_test/annotations_reconstructed.xml', 'w') as file:
        file.write(xml_string_reconstructed)


# # Convert the dictionary to a JSON string for pretty printing
# json_str = json.dumps(xml_dict, indent=4)
# print(json_str)
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# for mask in tqdm(masks):
#     img = mask_to_img(mask['rle'], mask['width'], mask['height'], mask['top'], mask['left'], (img_dim['height'], img_dim['width']))
    
#     plt.imshow(img)
#     rle, width, height, top, left, img_dim2 = img_to_mask(img)
#     if not np.array_equal(rle, mask['rle']):
#         print("RLE mismatch, rle: {} \n is not equal to mask rle: {}".format(rle, mask['rle']))
#     assert width == mask['width']
#     assert height == mask['height']
#     assert top == mask['top']
#     assert left == mask['left']
#     assert img_dim['height'] == img_dim2[0]
#     assert img_dim['width'] == img_dim2[1]