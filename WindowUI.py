from PyQt5.QtGui import QImage, QPixmap
from SketchGUI import Ui_SketchGUI
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QGraphicsItem, QColorDialog, QGraphicsView
import numpy as np
import cv2
from Input_mouse_event import InputGraphicsScene
from Output_mouse_event import OutputGraphicsScene
from time import gmtime, strftime
import time
import _thread as thread
import os
import jittor as jt

jt.flags.use_cuda = 1

class WindowUI(QtWidgets.QMainWindow,Ui_SketchGUI):

    def __init__(self):
        super(WindowUI, self).__init__()
        self.setupUi(self)
        self.setEvents()
        self._translate = QtCore.QCoreApplication.translate

        self.output_img = None
        self.brush_size = self.BrushSize.value()
        self.eraser_size = self.EraseSize.value()

        self.modes = [0,1,0] #0 marks the eraser, 1 marks the brush
        self.Modify_modes = [0,1,0] #0 marks the eraser, 1 marks the brush

        self.output_scene = OutputGraphicsScene()
        self.output.setScene(self.output_scene)
        self.output.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_view = QGraphicsView(self.output_scene)
        # self.output_view.fitInView(self.output_scene.updatePixmap())

        self.input_scene = InputGraphicsScene(self.modes, self.brush_size,self.output_scene)
        self.input.setScene(self.input_scene)
        self.input.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.input.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.input_scene.convert_on = self.RealTime_checkBox.isChecked()
        self.output_scene.convert_on = self.RealTime_checkBox.isChecked()

        self.BrushNum_label.setText(self._translate("SketchGUI", str(self.brush_size)))
        self.EraserNum_label.setText(self._translate("SketchGUI", str(self.eraser_size)))

        self.start_time = time.time()
        # self.


        # try:
        #     # thread.start_new_thread(self.output_scene.fresh_board,())
        #     thread.start_new_thread(self.input_scene.thread_shadow,())
        # except:
        #     print("Error: unable to start thread")
        # print("Finish")

    def setEvents(self):
        self.Undo_Button.clicked.connect(self.undo)

        self.Brush_Button.clicked.connect(self.brush_mode)
        self.BrushSize.valueChanged.connect(self.brush_change)

        self.Clear_Button.clicked.connect(self.clear)
      
        self.Eraser_Button.clicked.connect(self.eraser_mode)
        self.EraseSize.valueChanged.connect(self.eraser_change)

        self.Save_Button.clicked.connect(self.saveFile)

        #weight bar
        self.part0_Slider.valueChanged.connect(self.changePart)
        self.part1_Slider.valueChanged.connect(self.changePart)
        self.part2_Slider.valueChanged.connect(self.changePart)
        self.part3_Slider.valueChanged.connect(self.changePart)
        self.part4_Slider.valueChanged.connect(self.changePart)
        self.part5_Slider.valueChanged.connect(self.changAllPart)

        self.Load_Button.clicked.connect(self.open)

        self.Convert_Sketch.clicked.connect(self.convert)
        self.RealTime_checkBox.clicked.connect(self.convert_on)
        self.Shadow_checkBox.clicked.connect(self.shadow_on)

        self.Female_Button.clicked.connect(self.choose_Gender)
        self.Man_Button.clicked.connect(self.choose_Gender)

        self.actionSave.triggered.connect(self.saveFile)

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def brush_mode(self):
        self.mode_select(1)
        self.brush_change()
        self.statusBar().showMessage("Brush")

    def eraser_mode(self):
        self.mode_select(0)
        self.eraser_change()
        self.statusBar().showMessage("Eraser")

    def undo(self):
        self.input_scene.undo()
        self.output_scene.undo()


    def brush_change(self):
        self.brush_size = self.BrushSize.value()
        self.BrushNum_label.setText(self._translate("SketchGUI", str(self.brush_size)))
        if self.modes[1]:
            self.input_scene.paint_size = self.brush_size
            self.input_scene.paint_color = (0,0,0)
        self.statusBar().showMessage("Change Brush Size in ", self.brush_size)

    def eraser_change(self):
        self.eraser_size = self.EraseSize.value()
        self.EraserNum_label.setText(self._translate("SketchGUI", str(self.eraser_size)))
        if self.modes[0]:
            print( self.eraser_size)
            self.input_scene.paint_size = self.eraser_size
            self.input_scene.paint_color = (1,1,1)
        self.statusBar().showMessage("Change Eraser Size in ", self.eraser_size)

    def changePart(self):
        self.input_scene.part_weight['eye1'] = self.part0_Slider.value()/100
        self.input_scene.part_weight['eye2'] = self.part1_Slider.value()/100
        self.input_scene.part_weight['nose'] = self.part2_Slider.value()/100
        self.input_scene.part_weight['mouth'] = self.part3_Slider.value()/100
        self.input_scene.part_weight[''] = self.part4_Slider.value()/100
        self.input_scene.start_Shadow()
        # self.input_scene.updatePixmap()

    def changAllPart(self):
        value = self.part5_Slider.value()
        self.part0_Slider.setProperty("value", value)
        self.part1_Slider.setProperty("value", value)
        self.part2_Slider.setProperty("value", value)
        self.part3_Slider.setProperty("value", value)
        self.part4_Slider.setProperty("value", value)
        self.changePart()

    def clear(self):
        self.input_scene.reset()
        self.output_scene.reset()
        self.start_time = time.time()
        self.input_scene.start_Shadow()
        self.statusBar().showMessage("Clear Drawing Board")

    def convert(self):
        self.statusBar().showMessage("Press Convert")
        self.input_scene.convert_RGB()
        self.output_scene.updatePixmap()

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath(),"Images Files (*.*)") #jpg;*.jpeg;*.png
        if fileName:
            image = QPixmap(fileName)
            mat_img = cv2.imread(fileName)
            mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGB2BGR)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return

            # cv2.imshow('open',mat_img)
            self.input_scene.start_Shadow()
            self.input_scene.setSketchImag(mat_img)

    def saveFile(self):
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        file_dir = './saveImage/'+cur_time
        if not os.path.isdir(file_dir) :
            os.makedirs(file_dir)
        
        cv2.imwrite(file_dir+'/hand-draw.jpg',self.input_scene.sketch_img*255)
        cv2.imwrite(file_dir+'/colorized.jpg',cv2.cvtColor(self.output_scene.ori_img, cv2.COLOR_BGR2RGB))

        print(file_dir)


    def convert_on(self):
        # if self.RealTime_checkBox.isCheched():
        print( 'self.RealTime_checkBox',self.input_scene.convert_on)
        self.input_scene.convert_on = self.RealTime_checkBox.isChecked()
        self.output_scene.convert_on = self.RealTime_checkBox.isChecked()

    def shadow_on(self):
        _translate = QtCore.QCoreApplication.translate
        self.input_scene.shadow_on = not self.input_scene.shadow_on
        self.input_scene.updatePixmap()
        if self.input_scene.shadow_on:
            self.statusBar().showMessage("Shadow ON")
        else:
            self.statusBar().showMessage("Shadow OFF")


    def choose_Gender(self):
        if self.Female_Button.isChecked():
            self.input_scene.sex = 1
        else:
            self.input_scene.sex = 0
        self.input_scene.start_Shadow()