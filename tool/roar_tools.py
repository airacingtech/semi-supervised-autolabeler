import numpy as np
import xml.etree.ElementTree as ET

def hex_to_rgb(hex_color_str):
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
    """
    
    #blank image
    img = np.zeros(img_dim)
    #Decode run length encoding
    bitmap = np.concatenate([np.zeros(n) if i % 2 == 0 else np.ones(n) \
                    for i, n in enumerate(rle)]).reshape(height, width)
    #Fill image
    img[top:top+height, left:left+width] = bitmap
    return img

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
    labels -- List of dicts with name, color and id
        name -- Name of mask
        color -- Color of mask
        id -- ID of mask
    img_dim -- Dictionary with keys 'width', 'height'
    """
    # Find Root
    tree = ET.parse(filename)
    root = tree.getroot()
    
    # Find label information
    labels = []
    for l_id, label in enumerate(root.findall('.//label')):
        l_name = label.find('name').text
        l_color = label.find('color').text
        labels.append({'name':l_name, 'color': l_color, 'id': l_id})
    
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
    
    return masks, labels, img_dim