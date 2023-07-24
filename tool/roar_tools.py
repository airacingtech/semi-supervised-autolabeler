def hex_to_rgb(hex_color_str):
    if hex_color_str[0] == '#':
        hex_color_str = hex_color_str[1:]
    
    if len(hex_color_str) != 6:
        raise ValueError("Input string must be a 6 character hexadecimal color string. Example: '#FFFFFF'")
    
    return [int(hex_color_str[i:i+2], 16) for i in range(0, 6, 2)]

###Create Mask objects from annotations.xml file