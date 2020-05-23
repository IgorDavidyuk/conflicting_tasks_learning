import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fontbakery.utils import get_FamilyProto_Message
from utils import render_character
from freetype import Face
import cv2
import time

import cProfile, pstats, io
def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())
        return retval
    return inner

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


def render_one_font(font_path, char_set, target_path, pic_size=64):
    full_name = font_path.split('/')[-1].split('.')[0]
    font_folder = os.path.join(target_path, full_name)
    os.makedirs(font_folder, exist_ok=True)
    face = Face(font_path)
    for char in char_set:
        img = render_character(face, char, pic_size)
        cv2.imwrite(os.path.join(font_folder, f'{char}.jpg'), img)
        continue

# @profile
def save_rendered_glyphs(gf_dataframe, char_set, target_path, pic_size=64):
    os.makedirs(target_path, exist_ok=True)
    blacklist_indeces = []
    for i, row in tqdm(gf_dataframe.iterrows(), total=gf_dataframe.shape[0]):
        try:
            render_one_font(row.path, char_set, target_path, pic_size)
        except Exception as e:
            print('failed rendering font', row.name)
            print('with exception:', e)
            blacklist_indeces.append(i)
        continue
    gf_dataframe.drop(blacklist_indeces, inplace=True)
    return

def save_rendered_glyphs_mt(gf_dataframe, char_set, target_path, pic_size=64):
    from concurrent import futures #ThreadPoolExecutor
    os.makedirs(target_path, exist_ok=True)

    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures_dict = {executor.submit(render_one_font, row.path, char_set, target_path, pic_size): row \
                    for _, row in gf_dataframe.iterrows()}

        for future in tqdm(futures.as_completed(futures_dict), total=gf_dataframe.shape[0]):
            row = futures_dict[future]
            try:
                _ = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (row.name, exc))
            else:
                pass
                # print('%r page is %d bytes' % (url, len(data)))
    return


def save_rendered_glyphs_mp(gf_dataframe, char_set, target_path, pic_size=64):
    import fork_futures as futures
    from fork_futures import ForkPoolExecutor as ProcessPoolExecutor
    os.makedirs(target_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=4) as executor:
        args = gf_dataframe.path.values
        executor.map(render_one_font, args, [char_set]*len(args), [target_path]*len(args), [pic_size]*len(args))
    return
    


if __name__ == "__main__":
    ofl_path = './fonts/ofl/'
    fonts_data = parse_gf_metadata(ofl_path)
    # blacklist_fonts = ['Kumar One', 'Rubik'] # fonts with broken tabels
    # # removing blacklisted fonts from the dataframe
    # indeces_to_remove = False
    # for font_name in blacklist_fonts:
    #     indeces_to_remove += (fonts_data.name==font_name).values
    # fonts_data.drop(np.where(indeces_to_remove)[0], inplace=True)

    capital_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_set = capital_alphabet
    for char in 'OQMWIN':
        char_set = char_set.replace(char, '')
    
    # marker = time.time()
    save_rendered_glyphs(fonts_data, char_set, './rendered_set', 64)
    # print('finished in', time.time()-marker, 'seconds')