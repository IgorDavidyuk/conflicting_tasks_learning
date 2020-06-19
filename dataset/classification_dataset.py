import os
# import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from create_google_fonts_dataset import parse_gf_metadata, save_rendered_glyphs


class CharClassificationDataset(Dataset):

    def __init__(self, gf_dataframe, root_dir, charset=None, img_size=64, transform=None):
        '''
        Args:
            gf_dataframe (pd.DataFrame): Info from parsed google fonts metadata files.
            root_dir (string): Directory with all the images grooped in folders by font name.
            charset (string): A string contains all the unique chars used for each typeface
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.gf_dataframe = gf_dataframe
        self.root_dir = root_dir
        self.img_size = img_size

        if charset is not None:
            self.charset = str(charset)
        else:
            # construct letter set
            capital_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            char_set = capital_alphabet
            for char in 'OQMWIN': # removing problematic symbols
                char_set = char_set.replace(char, '')
            self.charset = char_set
        
        # render dataset if it is absent and check validity if it was rendered before
        self.render_fonts()
        self.font_names = list(map(self.split_path, self.gf_dataframe.path.values))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    # we use 1 channel images
                    transforms.Normalize(mean=[np.mean([0.485, 0.456, 0.406])],
                                                std=[np.mean([0.229, 0.224, 0.225])]) 
            ])        

    @staticmethod
    def split_path(path):
        return path.split('/')[-1].split('.')[0]
        
        
    def render_fonts(self):
        '''
        Check existans of pre-rendered dataset in root_dir
        '''
        for i, row in self.gf_dataframe.iterrows():
            font_name = row.path.split('/')[-1].split('.')[0]
            for char in self.charset:
                pic_address = os.path.join(self.root_dir, font_name, f'{char}.jpg')
                try:
                    size_b = os.path.getsize(pic_address)
                    if size_b > 100:
                        continue
                    else: raise Exception()
                except:
                    print(f'valid image {pic_address} is absent')
                    print(f'\ncreating dataset in {self.root_dir} directory\n')
                    save_rendered_glyphs(self.gf_dataframe, self.charset, self.root_dir, \
                            pic_size=self.img_size )
                    return
                

    def __len__(self):
        return len(self.gf_dataframe) * len(self.charset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # font_name = self.gf_dataframe.path[idx//len(self.charset)].split('/')[-1].split('.')[0]
        font_name = self.font_names[idx//len(self.charset)]
        letter_id = idx%len(self.charset)
        letter = self.charset[letter_id]
        img_name = os.path.join(self.root_dir, font_name, f'{letter}.jpg')
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, letter_id