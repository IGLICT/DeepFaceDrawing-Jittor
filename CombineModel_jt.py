# -*- coding: utf-8 -*-

import numpy as np
import cv2
from models.AE_Model import AE_Model
from models.Combine_Model import InferenceModel
from options.AE_face import wholeOptions
from options.parts_combine import CombineOptions
import _thread as thread
import time
import scipy.ndimage as sn
import random
import jittor as jt

class CombineModel():
    def __init__(self):
        self.sketch_img = np.ones((512, 512, 3),dtype=np.float32)
        self.generated = np.ones((512, 512, 3),dtype=np.uint8)*255

        self.sample_Num = 15

        self.inmodel=1

        self.fakes = {}
        self.mask = {}
        self.vector_part = {}
        self.shadow = {}
        self.shadow_on = True

        self.sex = 0

        # model
        if self.inmodel:
            #models for face/eye1/eye2/nose/mouth
            self.model = {'eye1':AE_Model(),
                          'eye2':AE_Model(),
                          'nose':AE_Model(),
                          'mouth':AE_Model(),
                          '': AE_Model()}
            self.part = {'eye1':(108,156,128),
                        'eye2':(255,156,128),
                        'nose':(182,232,160),
                        'mouth':(169,301,192),
                        '':(0,0,512)}
            self.opt = wholeOptions().parse(save=False)
            for key in  self.model.keys():
                # print(key)
                self.opt.partial = key
                self.model[key].initialize(self.opt)
                self.model[key].eval()

                self.mask[key] = cv2.cvtColor(cv2.imread('heat/' + key + '.jpg'), cv2.COLOR_RGB2GRAY).astype(np.float) / 255
                self.mask[key] = np.expand_dims(self.mask[key], axis=2)
                # print(key,'mask',self.mask[key].shape, self.mask[key].min())

            #for input and refine weight
            self.part_weight = {'eye1': 1,
                                'eye2': 1,
                                'nose': 1,
                                'mouth': 1,
                                '': 1}

        opt1 = CombineOptions().parse(save=False)
        opt1.nThreads = 1  # test code only supports nThreads = 1
        opt1.batchSize = 1  # test code only supports batchSize = 1

        self.combine_model = InferenceModel()
        self.combine_model.initialize(opt1)


        self.random_ = random.randint(0, self.model[''].feature_list[self.sex].shape[0])


    def predict_shadow(self, sketch_mat):
        width = 512
        # sketch = (self.sketch_img*255).astype(np.uint8)
        if self.inmodel:
            fake = {}
            shadow = {}
            vector_part = {}
            idx = 0
            for key in self.model.keys():
                loc = self.part[key]
                sketch_part = sketch_mat[loc[1]:loc[1]+loc[2],loc[0]:loc[0]+loc[2],:]

                if key == '' :
                    for key_p in self.model.keys():
                        if key_p!= '':
                            loc_p = self.part[key_p]
                            sketch_part[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:] = 255

                shadow_, vector_part[key] = self.model[key].get_inter(sketch_part[:, :, 0],
                                                                    self.sample_Num,
                                                                    w_c = self.part_weight[key],
                                                                    sex=self.sex)

                #shadow_ = shadow_[loc[1]:loc[1] + loc[2], loc[0]:loc[0] + loc[2], :]

                if key == '':
                    for key_p in self.model.keys():
                        if key_p != '':
                            loc_p = self.part[key_p]
                            shadow_[loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2], :] = 255 - (
                                        255 - shadow_[loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2], :]) * (1 - (
                                        1 - self.mask[key_p]) * 0.2)
                self.shadow[key] = np.ones((width, width, 1), dtype=np.uint8) * 255
                self.shadow[key][loc[1]:loc[1] + loc[2], loc[0]:loc[0] + loc[2], :] = 255 - (255 - shadow_) * (1 - self.mask[key])

                idx = idx+1
                
                jt.gc()
        else :
            fake = np.ones_like( sketch ) * 255
        # self.fakes = fake
        self.vector_part = vector_part
        self.shadow = shadow
        
        #print(vector_part.keys())
        self.generated = self.combine_model.inference(vector_part)
        jt.gc()
        