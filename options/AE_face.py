import argparse
import os

class wholeOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='AE_whole',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--partial', type=str, default='', help='facial part mouth/eye1/eye2/nose/')
        self.parser.add_argument('--latant_dim', type=int, default=512, help='vae latent dim')


        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='ae', help='which model to use')

        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        self.parser.add_argument('--param', type=str, default='./Params', help='models are saved here')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = False

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        args = vars(self.opt)

        return self.opt
