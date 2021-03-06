from freetype import Face
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

def render_character(free_type_face, char, pic_size=64):
    if not isinstance(free_type_face, Face):
        try: free_type_face = Face(free_type_face)
        except Exception as e:
            print('not a FreeType face nor a path to .ttf file' +  
                    'provided in variable free_type_face = {free_type_face}')
            print(e)
            return None
    pic_size = int(pic_size)
    # anti-aliasing
    free_type_face.set_pixel_sizes(3*pic_size, 0)
    try:
        free_type_face.load_char(str(char)[0])
    except Exception as e:
        print(f'failed to load character {char} from {str(free_type_face.family_name)}-{str(free_type_face.style_name)}')
        print(f'Exception: {e}')
        return None
    bitmap = free_type_face.glyph.bitmap
    img = np.array(bitmap.buffer).reshape(bitmap.rows,-1)
    h,w = img.shape
    # pad glyph to square
    if h >= w:
        left = (h-w)//2
        right = h - w - left
        top, bottom = 0, 0
    else:
        top = (w-h)//2
        bottom = w - h - top
        left, right = 0, 0
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    # resize
    return cv2.resize(img.astype('float32'), (pic_size, pic_size))

def visualize_set_of_chars(font_path, char_set, pic_size=64):
    '''
    Function draws glyphs of provided font
    input:
    font_path - path to a .ttf file
    pic_size - integer, width = height of 
    '''
    if not isinstance(char_set, str):
        print(f'char_set variable should contain a string, {type(char_set)} provided')
    char_set = str(char_set)
    pic_size = int(pic_size)
    face = Face(font_path)
    
    subplot_rows = math.ceil(len(char_set) / 10)
    _, axs = plt.subplots(subplot_rows, 10, figsize=(18,5))
    print(len(axs))
    for i in range(subplot_rows):
        for j in range(10):
            if 10*i + j < len(char_set):
                char = char_set[10*i + j]
                img = render_character(face, char, pic_size)
            else: 
                continue        
            axs[i][j].imshow(img)
        continue
    plt.tight_layout()
    plt.show()

def visualize_digits(font_path, pic_size):
    char_set = ['0123456789']
    visualize_set_of_chars(font_path, char_set, pic_size)
    
    
        


    

