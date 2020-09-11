import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import pytorch_lightning as pl
sys.path.insert(1, '../')
from dataset.classification_dataset import CharClassificationDataset
from dataset.create_google_fonts_dataset import parse_gf_metadata

class Classification_model(pl.LightningModule):
    # data preparation part
    def prepare_data(self):
        # this method is called once

        # 1) read google fonts metadata
        ofl_path = './dataset/fonts/ofl/'
        fonts_data = parse_gf_metadata(ofl_path) # google fonts dataframe
        # 2) removing blacklisted fonts from the dataframe
        # for some reason freetype can't read and render them
        blacklist_fonts = ['Kumar One', 'Rubik'] # fonts with broken tabels
        for font_name in blacklist_fonts:
            indeces_to_remove = (fonts_data.name==font_name).values
        fonts_data.drop(np.where(indeces_to_remove)[0], inplace=True)
        # 3) choose letter set
        # We use here capital letters excluding 'OQMWIN'
        # as they dont fit into square image.
        # This letter set is embedded into the dataset as default,
        # so there is no need to do this here
        '''
        capital_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        char_set = capital_alphabet
        for char in 'OQMWIN': # removing problematic symbols
            char_set = char_set.replace(char, '')
        '''
        # 4) compose a transform set to apply to images
        # This transform set is embedded into the dataset as default,
        # so there is no need to do this here
        '''
        tfs = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[np.mean([0.485, 0.456, 0.406])],
                                    std=[np.mean([0.229, 0.224, 0.225])]) 
        '''
        # 5) point to a directory where rendered images should be stored
        root_dir = './dataset/rendered_set/'
        # if we want to make test run on a fraction of the dataset
        #'''
        fonts_total = len(fonts_data)
        fonts_data = fonts_data[:int(fonts_total/4)]
        #'''
        # 6) fonts rendering itself is embedded into the dataset
        self.dataset = CharClassificationDataset(fonts_data, root_dir, \
                                          img_size=self.img_size)
        
    
    def setup(self, stage):
        # split the dataset
        data_size = len(self.dataset)
        val_split = int(np.floor((self.validation_fraction) * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        self.train_sampler = SubsetRandomSampler(indices[val_split:])
        self.val_sampler = SubsetRandomSampler(indices[:val_split])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                          sampler=self.train_sampler, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                          sampler=self.val_sampler, num_workers=4, pin_memory=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
        return [optimizer], [scheduler]


    def __init__(self, backbone, lr=2e-4, weight_decay=1e-3, in_c=1, out_c=20):
        super().__init__()
        # image size s to be used in training / 
        # should be devided without remnant by 32
        self.img_size = 64
        # batch settings
        self.batch_size = 70
        self.validation_fraction = .4

        self.backbone = backbone.requires_grad_(False)
        # rebuild model's input and output layers
        self.backbone = self.replace_first_conv_and_classifier_head(self.backbone, \
                                      in_channels=in_c, out_classes=out_c)

        # learning settings
        self.lr = lr
        self.weight_decay = weight_decay

        np.random.seed(4)

    def forward(self, x):
        return self.backbone.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return {'val_loss': val_loss}
      
    def training_epoch_end(self, outputs):
        # if outputs[-1]['train_loss'] < 1:
        self.backbone.requires_grad_(True)
        return dict()

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def replace_first_conv_and_classifier_head(model, in_channels, out_classes):
        # define helper functions
        # for name,_ in model.named_parameters():
        #   print(name)
        def recursive_gettatr(obj, attr_list):
            if len(attr_list) == 1:
                return getattr(obj, attr_list[0])
            else:
                return recursive_gettatr(obj=getattr(obj, attr_list[0]), attr_list=attr_list[1:])

        def recursive_settatr(obj, attr_list, value):
            if len(attr_list) == 1:
                return setattr(obj, attr_list[0], value)
            else:
                return recursive_settatr(getattr(obj, attr_list[0]), attr_list[1:], value)

        # search for the first convolution
        for name, _ in model.named_parameters():
            name = name.split('.')[:-1]
            first_layer = recursive_gettatr(model, name)
            if isinstance(first_layer, torch.nn.Conv2d):
                first_conv = first_layer
                break
        # replace first conv with custom conv
        out_c = first_conv.out_channels
        ker_s = first_conv.kernel_size
        stride = first_conv.stride
        pad = first_conv.padding
        dilation = first_conv.dilation
        recursive_settatr(model, name, torch.nn.Conv2d(in_channels, out_c, ker_s, stride, pad, dilation))

        # get classification head (the last layer)
        name, _ = list(model.named_parameters())[-1]
        name = name.split('.')[:-1]
        class_head = recursive_gettatr(model, name)
        in_f = class_head.in_features
        bias = class_head.bias is not None
        recursive_settatr(model, name, torch.nn.Linear(in_f, out_classes, bias))

        return model
