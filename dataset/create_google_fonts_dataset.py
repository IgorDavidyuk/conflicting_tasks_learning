import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fontbakery.utils import get_FamilyProto_Message

def parse_gf_metadata(ofl_path):
    '''
    Functions parses all the 'METADATA.pb' files in the Google Fonts repo 
    and ignores all non-latin fonts
    input:
    ofl_path - path to 'ofl' dir inside Google Fonts, usually fonts/ofl
    returns:
    a pandas dataframe containing parsed info - one row for one font
    formated as follows:
    [font name, path to font, category, style, weight, subsets]
    '''
    columns=['name', 'path', 'category', 'style', 'weight', 'subsets']
    parsed_metadata = []
    for dirpath, _, _ in os.walk(ofl_path):      
        metadataProtoFile = os.path.join(dirpath, 'METADATA.pb')
        if not os.path.exists(metadataProtoFile):
            continue
        try:
            message = get_FamilyProto_Message(metadataProtoFile)
        except Exception as e:
            print(metadataProtoFile)
            print(repr(e))
            continue
        subsets = message.subsets
        if 'latin' not in subsets:
            continue
        name = message.name
        cat = message.category
        for font in message.fonts:
            style = font.style
            weight = font.weight
            filename = font.filename
            font_line = [name, os.path.join(dirpath, filename), cat, style, weight, subsets]
            parsed_metadata.append(font_line)

    return pd.DataFrame(parsed_metadata, columns=columns)


def save_rendered_glyphs(gf_dataframe, char_set, target_path, pic_size=64):
    from utils import render_character
    from freetype import Face
    import cv2
    os.makedirs(target_path, exist_ok=True)
    for i, row in tqdm(gf_dataframe.iterrows(), total=gf_dataframe.shape[0]):
        path = row.path
        full_name = path.split('/')[-1].split('.')[0]
        font_folder = os.path.join(target_path, full_name)
        os.makedirs(font_folder, exist_ok=True)

        face = Face(path)
        for char in char_set:
            img = render_character(face, char, pic_size)
            cv2.imwrite(os.path.join(font_folder, f'{char}.jpg'), img)
            continue
        continue
    return
    

if __name__ == "__main__":
    ofl_path = './fonts/ofl/'
    fonts_data = parse_gf_metadata(ofl_path)
    blacklist_fonts = ['Kumar One', 'Rubik'] # fonts with broken tabels
    # removing blacklisted fonts from the dataframe
    indeces_to_remove = False
    for font_name in blacklist_fonts:
        indeces_to_remove += (fonts_data.name==font_name).values
    fonts_data.drop(np.where(indeces_to_remove)[0], inplace=True)

    capital_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_set = capital_alphabet
    for char in 'OQMWIN':
        char_set = char_set.replace(char, '')
    save_rendered_glyphs(fonts_data, char_set, './rendered_set', 64)