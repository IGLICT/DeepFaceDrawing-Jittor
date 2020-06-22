from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
import sys
from WindowUI import WindowUI
import os
import glob
from PyQt5 import QtGui,QtCore



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = WindowUI()
    ui.show()
    sys.exit(app.exec_())