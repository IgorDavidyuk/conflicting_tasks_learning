import os
import pandas as pd
from fontbakery.utils import get_FamilyProto_Message

def parse_google_fonts_metadata(ofl_path):
    # stores data in format [font name, path to font, category, style, weight, subsets]
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
        name = message.name
        cat = message.category
        subsets = message.subsets
        for font in message.fonts:
            style = font.style
            weight = font.weight
            filename = font.filename
            font_line = [name, os.path.join(dirpath, filename), cat, style, weight, subsets]
            parsed_metadata.append(font_line)


    return pd.DataFrame(parsed_metadata, columns=columns)


if __name__ == "__main__":
    ofl_path = './fonts/ofl/'
    fonts_data = parse_google_fonts_metadata(ofl_path)