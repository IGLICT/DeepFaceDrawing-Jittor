# this is model for fix netP and mask only left eye, right eye, nose and skin
# original loss

import random
import numpy as np
import os
from . import networks
# from scipy.ndimage import median_filter
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
# from pdb import set_trace as st
# import heapq
from numpy.linalg import solve
import time


class AE_Model(nn.Module):
    def name(self):
        return self.name

    def init_loss_filter(self):
        flags = (True,True,True)
        def loss_filter(kl_loss,l2_image,l2_mask):
            return [l for (l,f) in zip((kl_loss,l2_image,l2_mask),flags) if f]
        
        return loss_filter
    
    def initialize(self, opt):
        # assert opt.vae_encoder == True
        self.opt = opt
        self.save_dir = os.path.join(opt.param, opt.name) 

        self.name = 'AE_Model'
        # BaseModel.initialize(self, opt)
      
        input_nc = opt.input_nc

        self.output_nc = opt.output_nc
        self.input_nc = input_nc

        self.model_partial_name = opt.partial
        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        

        self.net_encoder = networks.define_part_encoder(model=self.model_partial_name, 
                                                                            input_nc = opt.input_nc, 
                                                                            norm=opt.norm, 
                                                                            latent_dim = opt.latant_dim)

        self.net_decoder = networks.define_part_decoder(model=self.model_partial_name, 
                                                                            output_nc = opt.input_nc, 
                                                                            norm=opt.norm, 
                                                                            latent_dim = opt.latant_dim)
        
        self.load_network(self.net_encoder, 'encoder_'+self.model_partial_name, 'latest', '')     
        self.load_network(self.net_decoder, 'decoder_'+self.model_partial_name+'_image', 'latest', '')   

        # use for test
        self.feature_list_male = np.fromfile(os.path.join(opt.param, opt.name)  + '/man_' + opt.partial + '_feature.bin', dtype=np.float32)
        self.feature_list_male.shape = 6247, 512


        #girl
        self.feature_list_female = np.fromfile(os.path.join(opt.param, opt.name)  + '/female_' + opt.partial + '_feature.bin', dtype=np.float32)
        self.feature_list_female.shape = 11456, 512
        self.feature_list = [self.feature_list_male,self.feature_list_female]

    def get_latent(self, input_image):

        input_image = (input_image-127.5)/127.5
        input_image = np.expand_dims(input_image, axis=2)
        input_image = input_image.transpose(2,0,1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = transform.to_tensor(jt.array(input_image))
        # print(input_image.shape)
        mus_mouth = self.net_encoder(input_image)

        return mus_mouth

    def get_image(self, latent_vec):

        # return self.net_decoder(latent_vec)

        fakes = self.net_decoder(latent_vec)
        fakes = (fakes[0,:,:,:].numpy()+1)/2

        fakes = np.transpose(fakes, (1, 2, 0)) * 255.0
        fakes = np.clip(fakes, 0, 255)

        return fakes.astype(np.uint8)

    def get_inter(self, input_image, nearnN=3, sex=1,w_c=1,random_=-1):
        generated_f = self.get_latent(input_image)
        generated_f = generated_f.numpy()
        
        feature_list = self.feature_list[sex]
        list_len = jt.array([feature_list.shape[0]])
        # a = jt.random((n,3))
        b = jt.code([1, nearnN], 
              "int32", [jt.array(feature_list),jt.array(generated_f), list_len], 
        cpu_header="#include <algorithm>",
        cpu_src="""
              using namespace std;
              auto n=out_shape0, k=out_shape1;
              int N=@in2(0);
              
              // 使用openmp实现自动并行化
                // 存储k近邻的距离和下标
                vector<pair<float,int>> id(N);
              #pragma omp parallel for
                for (int j=0; j<N; j++) {
                    auto dis = 0.0;
                    for (int d=0; d<512; d++)
                    {
                      auto dx = @in1(0,d)-@in0(j,d);
                      dis = dis +dx*dx;
                    }
                    id[j] = {dis, j};
                }
                // 使用c++算法库的nth_element排序
                nth_element(id.begin(), 
                  id.begin()+k, id.end());
                // 将下标输出到计图的变量中
                for (int j=0; j<k; j++)
                  @out(0,j) = id[j].second;
              """
        )

        idx_sort = b[0].numpy()

        if nearnN==1:
            vec_mu = feature_list[idx_sort[0]]
            vec_mu = vec_mu * w_c + (1 - w_c) * generated_f
            return self.get_image(vec_mu), self.get_shadow_image(vec_mu, torch.ones((1,1)).data.cuda(), nearnN), vec_mu

        # |  vg - sum( wi*vi )|   et. sum(wi) = 1
        # == | vg - v0 - sum( wi*vi) |   et. w = [1,w1,...,wn]
        A_0 = [feature_list[idx_sort[0],:]]
        A_m = A_0
        for i in range(1,nearnN):
            A_m = np.concatenate((A_m,[feature_list[idx_sort[i],:]]), axis=0)
        
        A_0 = np.array(A_0)
        A_m= np.array(A_m).T
        A_m0 = np.concatenate((A_m[:,1:]-A_0.T, np.ones((1,nearnN-1))*10), axis=0)

        A = np.dot(A_m0.T, A_m0)
        b = np.zeros((1, generated_f.shape[1]+1))
        b[0,0:generated_f.shape[1]] = generated_f-A_0

        B = np.dot(A_m0.T, b.T)

        x = solve(A, B)

        xx = np.zeros((nearnN,1))
        xx[0,0] = 1 - x.sum()
        xx[1:,0] = x[:,0]
        # print(time.time()- start_time)

        vec_mu = np.dot(A_m, xx).T * w_c + (1-w_c)* generated_f
        vec_mu = jt.array(vec_mu.astype('float32'))

        return self.get_shadow_image(A_m.T,xx,nearnN), vec_mu

    def get_shadow_image(self, mus_mouth, weight, nearnN):

        fakes = 0
        for i in range(nearnN):
            w_i = weight[i]
            if w_i<=0:
                continue
            elif w_i>0.5:
                w_i = 0.5

            # print(i)
            mus_vec = jt.unsqueeze(mus_mouth[[i],:],1)

            fake_image = self.net_decoder(jt.array(mus_vec))
            # fake_image = fake_image[[0],:,:,:]
            if i==0:
                fakes = (1-fake_image)/2* w_i
            else:
                fakes = fakes + (1-fake_image)/2 * w_i

        fakes = 1-fakes

        fakes = fakes[0,:,:,:].detach().numpy()

        fakes = np.transpose(fakes, (1, 2, 0)) * 255.0
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