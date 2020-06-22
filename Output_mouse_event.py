# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import cv2
import _thread as thread
import time

class OutputGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        # self.modes = mode_list
        self.mouse_clicked = False
        self.prev_pt = None
        self.setSceneRect(0,0,self.width(),self.height())
        # self.masked_image = None

        self.selectMode = 0

        # save the history of edit
        self.history = []

        self.ori_img = np.ones((512, 512, 3),dtype=np.uint8)*255

        self.mask_put = 1 # 1 marks use brush while 0 user erase
        self.convert = False
        # self.setPos(0 ,0)
        self.firstDisplay = True
        self.convert_on = False
        
    def reset(self):

        self.convert = False
        self.ori_img = np.ones((512, 512, 3),dtype=np.uint8)*255
        self.updatePixmap(True)

        self.prev_pt = None

    def setSketchImag(self, sketch_mat, mouse_up=False):

        self.ori_img = sketch_mat.copy()

        self.image_list = []
        self.image_list.append( self.ori_img.copy() )

    def mousePressEvent(self, event):
        if not self.mask_put or self.selectMode == 1:
            self.mouse_clicked = True
            self.prev_pt = None
        else:
            self.make_sketch(event.scenePos())

    def make_sketch_Eraser(self, pts):
        if len(pts)>0:
            for pt in pts:
                cv2.line(self.color_img,pt['prev'],pt['curr'],self.paint_color,self.paint_size )
                cv2.line(self.mask_img,pt['prev'],pt['curr'],(0,0,0),self.paint_size )

        self.updatePixmap()

    def modify_sketch(self, pts):
        if len(pts)>0:
            for pt in pts:
                cv2.line(self.ori_img,pt['prev'],pt['curr'],self.paint_color,self.paint_size )
        self.updatePixmap()


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
            self.ori_img = self.image_list[num].copy()
            self.image_list.pop(num+1)

        self.updatePixmap(True)

    def getImage(self):
        return self.ori_img*(1-self.mask_img)  + self.color_img*self.mask_img


    def updatePixmap(self,mouse_up=False):

        sketch = self.ori_img
        qim = QImage(sketch.data, sketch.shape[1], sketch.shape[0], QImage.Format_RGB888)

        if self.firstDisplay :
            self.reset_items()
            self.imItem = self.addPixmap(QPixmap.fromImage(qim))
            self.firstDispla = False
        else:
            self.imItem.setPixmap(QPixmap.fromImage(qim))

    def fresh_board(self):
        print('======================================================')
        while(True):
            if(self.convert_on):
                print('======================================================')
                time.sleep(100)
                iter_start_time = time.time()
                self.updatePixmap()
                print('Time Sketch:',time.time() - iter_start_time)