import numpy as np
import os
from . import networks

import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform

class Combine_Model(nn.Module):
    def name(self):
        return 'Combine_Model'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.param, opt.name) 
        # BaseModel.initialize(self, opt)
        input_nc = opt.input_nc

        ##### define networks        
        # Generator network       
        self.part = {'': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}


        self.Decoder_Part = {}

        for key in self.part.keys():
            self.Decoder_Part[key] = networks.define_feature_decoder(model=key, 
                                                                output_nc = 32, norm=opt.norm, 
                                                                latent_dim = opt.latant_dim)
           
        self.netG = networks.define_G(opt.num_inter_channels, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.norm)
        
        self.load_network(self.netG, 'G', opt.which_epoch, '')
            
        for key in self.part.keys():
            self.load_network(self.Decoder_Part[key], 'DE_'+key, opt.which_epoch, '')

    def inference(self, part_v, image=None):

        eye1_code = part_v['eye1']
        eye2_code = part_v['eye2']
        nose_code = part_v['nose']
        mouth_code = part_v['mouth']
        bg_code = part_v['']
        
        eye1_r_feature = self.Decoder_Part['eye1'](eye1_code)
        eye2_r_feature = self.Decoder_Part['eye2'](eye2_code)
        nose_r_feature = self.Decoder_Part['nose'](nose_code)
        mouth_r_feature = self.Decoder_Part['mouth'](mouth_code)
        bg_r_feature = self.Decoder_Part[''](bg_code)

        bg_r_feature[:, :, 301:301 + 192, 169:169 + 192] = mouth_r_feature
        bg_r_feature[:, :, 232:232 + 160 - 36, 182:182 + 160] = nose_r_feature[:, :, :-36, :]
        bg_r_feature[:, :, 156:156 + 128, 108:108 + 128] = eye1_r_feature
        bg_r_feature[:, :, 156:156 + 128, 255:255 + 128] = eye2_r_feature

        input_concat = bg_r_feature      
           
        fake_image = self.netG(input_concat)

        # fakes = fake_image.detach().numpy()
        fakes = fake_image[0, :, :, :].detach().numpy()

        fakes = (np.transpose(fakes, (1, 2, 0)) + 1) / 2.0 * 255.0
        fakes = np.clip(fakes, 0, 255)

        return fakes.astype(np.uint8)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir='', save_path=''):        
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)    
        print("load_path",save_path)    
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            network.load(save_path) 

class InferenceModel(Combine_Model):
    def forward(self, inp):
        label, image = inp
        return self.inference(label, image)
