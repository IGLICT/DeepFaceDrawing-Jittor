import argparse
import os

class CombineOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='Combine',
                                 help='name of folder')
        self.parser.add_argument('--param', type=str, default='./Params', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for generator
        self.parser.add_argument('--ngf', type=int, default=56, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')

        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--latant_dim', type=int, default=512, help='vae latent dim') 
        self.parser.add_argument('--num_inter_channels', default=32, help='# of channels of the intermediate feature maps')
        
        self.isTrain = False

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        return self.opt