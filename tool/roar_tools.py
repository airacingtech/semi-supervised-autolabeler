import numpy as np
import xml.etree.ElementTree as ET
import datumaro
import ijson
from collections import defaultdict
# from RoarSegTracker import RoarSegTracker

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

class MaskObject():
    """MaskObject class representation
    """
    def __init__(self, frame: int, mask: dict, id: int, label="", color_id="", data={}, img_dim = (0, 0)):
            self.frame = frame
            self.mask = mask
            self.id = id
            self.label = label
            self.color = color_id
            self.data = data
            self.img_dim = img_dim
            self.start_0 = False #sometimes converting img back to xml parameters doesnt work if rle starts with 0
    
    def get_frame(self) -> int:
        return self.frame
    
    def get_mask(self) -> dict:
            return self.mask
    def get_id(self) -> int:
        return self.id
    def get_label(self) -> str:
            return self.label
    def get_color(self):
        return self.color
            
    def get_mask_array(self) -> np.array:
        """Get mask as numpy array.
        
        Returns:
        mask_array -- Numpy array of dimensions (height, width) with mask as 1 and background as 0.
        """
        mask_array = mask_to_img(self.data['rle'], self.data['width'], self.data['height'], 
                                 self.data['top'], self.data['left'], self.img_dim)
        
        return mask_array
    def turn_mask_array_to_dict(self, mask_array):
        self.data['rle'], self.data['width'], self.data['height'], \
        self.data['top'], self.data['left'], self.img_dim = img_to_mask(mask_array)

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
    img = np.zeros(img_dim)
    #Decode run length encoding
    bitmap = np.concatenate([np.zeros(n) if i % 2 == 0 else np.ones(n) \
                    for i, n in enumerate(rle)]).reshape(height, width)
    #Fill image
    img[top:top+height, left:left+width] = bitmap
    
    return img, rle[0] == 0

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
    
    # Find the bounding box of the mask
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract the mask and its dimensions
    mask = img[rmin:rmax+1, cmin:cmax+1]
    height, width = mask.shape
    
    # Create the run length encoding
    rle = []
    current_value = mask[0,0]
    current_run = 1
    for value in mask.flatten()[1:]:
        if value == current_value:
            current_run += 1
        else:
            rle.append(current_run)
            current_run = 1
            current_value = value
    rle.append(current_run)
    
    return (np.array(rle), width, height, rmin, cmin, img.shape)

def xml_to_masks(filename: str, use_labels_dict=False):
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
    labels -- List of dicts with name, color and id if use_labels_dict = False
        name -- Name of mask
        color -- Color of mask
        id -- ID of mask
    labels_dict -- Dictionary with keys 'label' if use_labels_dict = True
        key -- 'label'
        value -- {name: Name of mask, color: Color of mask, id: ID of mask}
    img_dim -- Dictionary with keys 'width', 'height'
    """
    # Find Root
    tree = ET.parse(filename)
    root = tree.getroot()
    
    # Find label information
    labels = []
    labels_dict = {}
    for l_id, label in enumerate(root.findall('.//label')):
        l_name = label.find('name').text
        l_color = label.find('color').text
        if not use_labels_dict:
            labels.append({'name':l_name, 'color': l_color, 'id': l_id})
        else:
            labels_dict['l_name'] = {'name':l_name, 'color': l_color, 'id': l_id}
    
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
    
    return masks, labels if not use_labels_dict else labels_dict, img_dim
def masks_to_mask_objects(masks: list, labels_dict: dict, img_dim: dict) -> dict[int, MaskObject]:
    """Gathers mask objects from a list of masks and return list of mask objects

    Args:
        masks (list): List of mask dictionaries
        labels_dict (dict): dictionary of labels to color and mask id
        img_dim (dict): dictionary containing dimensions of image 'height' and 'width'

    Returns:
        dict[int, MaskObject]: Dictionary pairing key frame to dict of mask objects mapped by object id
        for that given frame
    """
    frame_to_mask_objects = {}
    for mask in masks:
        mask_obj = MaskObject(mask['frame'], mask['rle'], mask['id'], labels_dict[mask['label']]['name'], 
                              labels_dict[mask['label']]['color'], mask, (img_dim['height'], img_dim['width']))
        if frame_to_mask_objects.get(mask_obj.frame) is None:
            frame_to_mask_objects[mask_obj.frame] = [{mask_obj.get_id(): mask_obj}]
        else:
            mask_dict = frame_to_mask_objects[mask_obj.frame]
            mask_dict[mask_obj.get_id()] = mask_obj
            
    return frame_to_mask_objects
import json

##TESTING
masks, labels, img_dim = xml_to_masks("/home/roar-nexus/Downloads/annotations.xml")
# roarseg = RoarSegTracker(masks, labels, img_dim)
import xmltodict
import json

# Parse the XML file into a Python dictionary
with open("/home/roar-nexus/Downloads/reupload_cvat_test/annotations.xml", 'r') as file:
    xml_string = file.read()
xml_dict = xmltodict.parse(xml_string)

# Convert the dictionary to a JSON string for pretty printing
json_str = json.dumps(xml_dict, indent=4)
print(json_str)

# Convert the dictionary back to an XML string
xml_string_reconstructed = xmltodict.unparse(xml_dict, pretty=True)
print(xml_string_reconstructed)

# Save the reconstructed XML to a file
with open('/home/roar-nexus/Downloads/reupload_cvat_test/annotations_reconstructed.xml', 'w') as file:
    file.write(xml_string_reconstructed)


# Convert the dictionary to a JSON string for pretty printing
json_str = json.dumps(xml_dict, indent=4)
print(json_str)
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