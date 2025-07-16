import colorsys

def create_unique_color_float(tag, hue_step=0.41):
    """
    Create a unique RGB color code for a given vehicle ID.
    
    Parameters
    ----------
    tag: int
        Vehicle ID to generate color for
    hue_step: float
        Step size for hue variation
        
    Returns
    -------
    tuple: (r, g, b) floats in range [0, 1]
    """
    h = (tag * hue_step) % 1.0
    s = 0.9  # High saturation for vehicle tracking
    v = 0.95  # High value for visibility
    return colorsys.hsv_to_rgb(h, s, v)

def create_unique_color(tag, hue_step=0.41):
    """
    Create a unique RGB color code for a given vehicle ID.
    
    Parameters
    ----------
    tag: int
        Vehicle ID to generate color for
    hue_step: float
        Step size for hue variation
        
    Returns
    -------
    tuple: (r, g, b) integers in range [0, 255]
    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)
