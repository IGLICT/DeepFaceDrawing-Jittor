# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
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

class InputGraphicsScene(QGraphicsScene):
    def __init__(self, mode_list, paint_size, up_sketch_view , parent=None):
        QGraphicsScene.__init__(self, parent)
        self.modes = mode_list
        self.mouse_clicked = False
        self.prev_pt = None
        self.setSceneRect(0,0,self.width(),self.height())

        # save the points
        self.mask_points = []
        self.sketch_points = []
        self.stroke_points = []
        self.image_list = []

        self.sex = 1
        self.up_sketch_view = up_sketch_view

        # save the history of edit
        self.history = []
        self.sample_Num = 15
        self.refine = True

        self.sketch_img = np.ones((512, 512, 3),dtype=np.float32)
        self.ori_img = np.ones((512, 512, 3),dtype=np.uint8)*255
        self.image_list.append( self.sketch_img.copy() )
        self.generated = np.ones((512, 512, 3),dtype=np.uint8)*255
        # strokes color
        self.stk_color = None
        self.paint_size = paint_size
        self.paint_color = (0,0,0)
        # self.setPos(0 ,0)

        self.inmodel=1

        self.mask = {}
        self.vector_part = {}
        self.shadow = {}
        self.shadow_on = True
        self.convert_on = False
        self.mouse_up = False


        # model
        if self.inmodel:
            #models for face/eye1/eye2/nose/mouth
            self.model = {}
            #crop location
            self.part = {'eye1':(108,156,128),
                        'eye2':(255,156,128),
                        'nose':(182,232,160),
                        'mouth':(169,301,192),
                        '':(0,0,512)}
            self.opt = wholeOptions().parse(save=False)
            for key in  self.part.keys():
                # print(key)
                self.opt.partial = key
                self.model[key] = AE_Model()
                self.model[key].initialize(self.opt)
                self.model[key].eval()

                self.mask[key] = cv2.cvtColor(cv2.imread('heat/' + key + '.jpg'), cv2.COLOR_RGB2GRAY).astype(np.float) / 255
                self.mask[key] = np.expand_dims(self.mask[key], axis=2)
               
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
        self.combine_model.eval()

        self.black_value = 0.0

        self.iter = 0
        self.max_iter = 20
        self.firstDisplay = True
        self.mouse_released = False

        self.random_ = random.randint(0, self.model[''].feature_list[self.sex].shape[0])

        self.predict_shadow()
        self.updatePixmap(True)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updatePixmap)
        self.timer.start(10)

    def reset(self):
        # save the points
        self.mask_points = []
        self.sketch_points = []
        self.stroke_points = []

        self.sketch_img = np.ones((512, 512, 3),dtype=np.float32)
        self.ori_img = np.ones((512, 512, 3),dtype=np.uint8)*255
        self.generated = np.ones((512, 512, 3),dtype=np.uint8)*255

        # save the history of edit
        self.history = []
        self.image_list.clear()
        self.image_list.append( self.sketch_img.copy() )

        self.updatePixmap(True)
        self.convert_RGB()

        self.prev_pt = None

        self.random_ = random.randint(0, self.model[''].feature_list[self.sex].shape[0])

    def setSketchImag(self, sketch_mat):

        self.reset()
        self.sketch_img = sketch_mat.astype(np.float32) / 255
        # self.sketch_img = sketch_mat
        self.updatePixmap()
        self.image_list.clear()
        self.image_list.append( self.sketch_img.copy() )

    def mousePressEvent(self, event):
        self.mouse_clicked = True
        self.prev_pt = None
        self.draw = False

    def mouseReleaseEvent(self, event):
        # print('Leave')
        self.start_Shadow()
        if self.draw :
            self.image_list.append(self.sketch_img.copy())
            self.updatePixmap(True)

        self.draw = False
        self.prev_pt = None
        self.mouse_clicked = False
        self.mouse_released = True
        self.mouse_up = True

    def mouseMoveEvent(self, event):
        if self.mouse_clicked:
            if int(event.scenePos().x())<0 or int(event.scenePos().x())>512 or int(event.scenePos().y())<0 or   int(event.scenePos().y())>512:
                return
            if self.prev_pt and int(event.scenePos().x()) == self.prev_pt.x()  and int(event.scenePos().y()) == self.prev_pt.y():
                return
            if self.prev_pt :
                # self.drawSketch(self.prev_pt, event.scenePos())
                pts = {}
                pts['prev'] = (int(self.prev_pt.x()),int(self.prev_pt.y()))
                pts['curr'] = (int(event.scenePos().x()),int(event.scenePos().y()))
                # self.sketch_points.append(pts)
                self.make_sketch( [pts])
                # self.history.append(1)
                self.prev_pt = event.scenePos()
            else:
                self.prev_pt = event.scenePos()

    def make_sketch(self, pts):
        if len(pts)>0:
            for pt in pts:
                cv2.line(self.sketch_img,pt['prev'],pt['curr'],self.paint_color,self.paint_size )
        self.updatePixmap()
        self.draw = True
        self.iter = self.iter+1
        if self.iter>self.max_iter:
            self.iter = 0

    def get_stk_color(self, color):
        self.stk_color = color

    def erase_prev_pt(self):
        self.prev_pt = None

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)

    def undo(self):
        if len(self.image_list)>1:
            num = len(self.image_list)-2
            self.sketch_img = self.image_list[num].copy()
            self.image_list.pop(num+1)

        self.updatePixmap(True)

    def getImage(self):
        return (self.sketch_img * self.ori_img).astype(np.uint8)

    def updatePixmap(self, mouse_up = False):
        # print('update')
        self.mouse_released = False

        #combine shadow
        shadow = self.shadow
        width = 512
        shadows = np.zeros((width, width, 1))
        for key in self.model.keys():
            if key == '':
                shadows = shadows + (255 - shadow[key])
            else:
                shadows = shadows + (255 - shadow[key]) * 0.5

        shadows = np.clip(shadows, 0, 255)

        self.ori_img = 255 - shadows * 0.4

        if self.shadow_on :
            sketch = (self.sketch_img * self.ori_img).astype(np.uint8)
        else:
            sketch = (self.sketch_img *255).astype(np.uint8)

        qim = QImage(sketch.data, sketch.shape[1], sketch.shape[0], QImage.Format_RGB888)
        if self.firstDisplay :
            self.reset_items()
            self.imItem = self.addPixmap(QPixmap.fromImage(qim))
            self.firstDispla = False
        else:
            self.imItem.setPixmap(QPixmap.fromImage(qim))

        if self.convert_on:
            self.convert_RGB()
            self.up_sketch_view.updatePixmap()

    def convert_RGB(self):
        self.up_sketch_view.setSketchImag(self.generated, True)

    def predict_shadow(self):
        width = 512
        sketch = (self.sketch_img*255).astype(np.uint8)
        if self.inmodel:
            shadow = {}
            vector_part = {}

            for key in self.model.keys():
                loc = self.part[key]
                sketch_part = sketch[loc[1]:loc[1]+loc[2],loc[0]:loc[0]+loc[2],:]

                if key == '' and self.refine:
                    for key_p in self.model.keys():
                        if key_p!= '':
                            loc_p = self.part[key_p]
                            sketch_part[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:] = 255

                # print(self.sex)
                if ((255-sketch_part).sum()==0):
                    shadow_, vector_part[key] = self.model[key].get_inter(sketch_part[:, :, 0],
                                                                                       self.sample_Num,
                                                                                       w_c = self.part_weight[key],
                                                                                       random_=self.random_,
                                                                                       sex=self.sex)
                else:
                    shadow_, vector_part[key] = self.model[key].get_inter(sketch_part[:, :, 0],
                                                                                       self.sample_Num,
                                                                                       w_c = self.part_weight[key],
                                                                                       sex=self.sex)

                if key == '':
                    for key_p in self.model.keys():
                        if key_p!= '':
                            loc_p = self.part[key_p]
                            shadow_[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:] = 255-(255-shadow_[loc_p[1]:loc_p[1]+loc_p[2],loc_p[0]:loc_p[0]+loc_p[2],:]) * (1-(1-self.mask[key_p])*0.2)
                shadow[key] = np.ones((width, width,1),dtype=np.uint8)*255
                shadow[key][loc[1]:loc[1]+loc[2],loc[0]:loc[0]+loc[2],:] = 255-(255-shadow_ )* (1 - self.mask[key])

        self.vector_part = vector_part
        self.shadow = shadow

    def start_Shadow(self):

        iter_start_time = time.time()
        self.predict_shadow()

        # iter_start_time = time.time()
        self.generated = self.combine_model.inference(self.vector_part)
        self.convert_RGB()
        self.updatePixmap()
        print('Time',time.time() - iter_start_time)

    def thread_shadow(self):
        while True:
            self.start_Shadow();