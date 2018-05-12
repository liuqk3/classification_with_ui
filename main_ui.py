# coding: utf-8

import sys
from PyQt5.QtWidgets import (QWidget, QLabel, QApplication, QFileDialog, QGridLayout, QLineEdit, QPushButton)
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import *
import time
from PIL import Image
import threading
from utils.cnn_functions import inference
import os
import signal
import subprocess


class ClassificationUI(QWidget):
    def __init__(self):
        super().__init__()

        self.show_log = False

        # create a palette, used to set color
        self.pe = QPalette()

        """ UI for training  """
        choose_train_data = QPushButton('choose train data')
        self.start_train = QPushButton('Start training')
        self.stop_train = QPushButton('Stop training')

        choose_train_data.clicked.connect(self.show_train_dialog)
        self.start_train.clicked.connect(self.start_train_func)
        self.stop_train.clicked.connect(self.stop_train_func)

        # used to show the chosen training directory
        self.train_dir_edit = QLineEdit()

        # used to show training statics
        self.statics_label =  QLabel('')
        self.statics_label.setAlignment(Qt.AlignTop)
        self.statics_label.setAutoFillBackground(True)
        self.pe.setColor(QPalette.Background, Qt.white)
        self.statics_label.setPalette(self.pe)

        self.loss_label = QLabel('')
        self.pe.setColor(QPalette.Background, Qt.white)
        self.loss_label.setAutoFillBackground(True)
        self.loss_label.setPalette(self.pe)
        self.resize_image(path="./buffer/blank.jpg", mode='loss')
        loss_im = QPixmap("./buffer/loss_buffer.jpg")
        loss_im = loss_im.scaledToWidth(430)
        loss_im = loss_im.scaledToHeight(300)
        self.loss_label.setPixmap(loss_im)

        # add buttons, label, and edit to the ui
        grid_train = QGridLayout()
        grid_train.setSpacing(10)

        grid_train.addWidget(choose_train_data, 1, 0)
        grid_train.addWidget(self.train_dir_edit, 1, 1, 1, 3)
        grid_train.addWidget(self.start_train, 1, 4, 1, 2)
        grid_train.addWidget(self.stop_train, 1, 6, 1, 2)
        grid_train.addWidget(self.statics_label, 2, 0, -1, 4)
        grid_train.addWidget(self.loss_label, 2, 4, -1, 4)

        """ UI for testing """
        choose_test_data = QPushButton('choose test data')
        choose_test_data.clicked.connect(self.show_test_dialog)

        self.test_dir_edit = QLineEdit()

        # create a label to show image
        self.image_label = QLabel()
        self.resize_image(path="./buffer/blank.jpg", mode='test')
        test_im = QPixmap("./buffer/test_buffer.jpg")
        test_im = test_im.scaledToWidth(430)
        test_im = test_im.scaledToHeight(430)
        self.image_label.setPixmap(test_im)

        self.results_label = QLabel('Results')
        self.results_label.setAlignment(Qt.AlignCenter)
        self.pe.setColor(QPalette.Background, Qt.lightGray)
        self.results_label.setAutoFillBackground(True)
        self.results_label.setPalette(self.pe)

        # add thses buttons, labels, edit to the ui
        grid_test = QGridLayout()
        grid_test.setSpacing(10)

        grid_test.addWidget(choose_test_data, 1, 0)
        grid_test.addWidget(self.test_dir_edit, 1, 1, 1, 3)
        grid_test.addWidget(self.image_label, 2, 0, -1, 4)
        grid_test.addWidget(self.results_label, 1, 4, 1, 4)

        # create ui for test results
        self.pe.setColor(QPalette.Background, Qt.white)

        collar_label = QLabel('领子设计')
        collar_label.setAlignment(Qt.AlignCenter)
        self.collar_result = QLabel('')
        self.collar_result.setAutoFillBackground(True)
        self.collar_result.setPalette(self.pe)
        self.collar_result.setAlignment(Qt.AlignCenter)

        neckline_label = QLabel('颈线设计')

        neckline_label.setAlignment(Qt.AlignCenter)
        self.neckline_result = QLabel('')
        self.neckline_result.setAutoFillBackground(True)
        self.neckline_result.setPalette(self.pe)
        self.neckline_result.setAlignment(Qt.AlignCenter)

        neck_label = QLabel('脖颈设计')
        neck_label.setAlignment(Qt.AlignCenter)
        self.neck_result = QLabel('')
        self.neck_result.setAutoFillBackground(True)
        self.neck_result.setPalette(self.pe)
        self.neck_result.setAlignment(Qt.AlignCenter)

        lapel_label = QLabel('翻领设计')
        lapel_label.setAlignment(Qt.AlignCenter)
        self.lapel_result = QLabel('')
        self.lapel_result.setAutoFillBackground(True)
        self.lapel_result.setPalette(self.pe)
        self.lapel_result.setAlignment(Qt.AlignCenter)

        coat_label = QLabel('衣长')
        coat_label.setAlignment(Qt.AlignCenter)
        self.coat_result = QLabel('')
        self.coat_result.setAutoFillBackground(True)
        self.coat_result.setPalette(self.pe)
        self.coat_result.setAlignment(Qt.AlignCenter)

        pant_laebl = QLabel('裤长')
        pant_laebl.setAlignment(Qt.AlignCenter)
        self.pant_result = QLabel('')
        self.pant_result.setAutoFillBackground(True)
        self.pant_result.setPalette(self.pe)
        self.pant_result.setAlignment(Qt.AlignCenter)

        skirt_label = QLabel('裙长')
        skirt_label.setAlignment(Qt.AlignCenter)
        self.skirt_result = QLabel('')
        self.skirt_result.setAutoFillBackground(True)
        self.skirt_result.setPalette(self.pe)
        self.skirt_result.setAlignment(Qt.AlignCenter)

        sleeve_label = QLabel('袖长')
        sleeve_label.setAlignment(Qt.AlignCenter)
        self.sleeve_result = QLabel('')
        self.sleeve_result.setAutoFillBackground(True)
        self.sleeve_result.setPalette(self.pe)
        self.sleeve_result.setAlignment(Qt.AlignCenter)

        grid_test.addWidget(collar_label, 2, 4, 1, 2)
        grid_test.addWidget(self.collar_result, 3, 4, 1, 2)
        grid_test.addWidget(neckline_label, 2, 6 ,1, 2)
        grid_test.addWidget(self.neckline_result, 3, 6, 1, 2)
        grid_test.addWidget(neck_label, 4, 4 , 1, 2)
        grid_test.addWidget(self.neck_result, 5, 4 ,1 ,2)
        grid_test.addWidget(lapel_label, 4, 6, 1, 2)
        grid_test.addWidget(self.lapel_result, 5, 6 , 1, 2)
        grid_test.addWidget(coat_label, 6, 4 , 1, 2)
        grid_test.addWidget(self.coat_result, 7, 4, 1, 2)
        grid_test.addWidget(pant_laebl, 6, 6, 1, 2)
        grid_test.addWidget(self.pant_result, 7, 6, 1, 2)
        grid_test.addWidget(skirt_label, 8, 4, 1, 2)
        grid_test.addWidget(self.skirt_result, 9, 4, 1, 2)
        grid_test.addWidget(sleeve_label, 8, 6, 1, 2)
        grid_test.addWidget(self.sleeve_result, 9, 6, 1, 2)

        """ combine two grid """
        upper_grid = QGridLayout()
        upper_grid.setSpacing(10)

        upper_grid.addLayout(grid_train, 1, 0)
        upper_grid.addLayout(grid_test, 2, 0)

        """ show the ui """
        self.setLayout(upper_grid)
        self.setGeometry(300, 100, 890, 860)
        self.setWindowTitle('image classification')
        self.setFixedSize(890, 860)
        self.show()

    def show_test_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/data0/yangwf/FashionAI/base/Images')[0]
        self.test_dir_edit.setText(fname)

        self.while_testing()

        self.test_thread = threading.Thread(target=self.make_prediction, name='test_thread')
        self.test_thread.start()

    def fill_log(self):
        while True:
            if not self.show_log:
                break
            time.sleep(1) # flash every 0.5 seconds
            if not os.path.exists('./buffer/loss.jpg'):
                continue
            else:

                self.resize_image(path='./buffer/loss.jpg', mode='loss')
                loss_im = QPixmap("./buffer/loss_buffer.jpg")
                loss_im = loss_im.scaledToWidth(430)
                loss_im = loss_im.scaledToHeight(300)
                self.loss_label.setPixmap(loss_im)
                # print('exist loss buffer jpg')

            if not os.path.exists("./buffer/log.log"):
                continue
            else:
                content = open("./buffer/log.log").readlines()
                if len(content) > 11:
                    content = content[-11:]
                content = "\n".join(content)
                self.statics_label.setText(content)


    def show_train_dialog(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open file', '/data0/yangwf/')
        self.train_dir_edit.setText(fname)
        return

    def start_train_func(self):

        # add the train data path
        fname = self.train_dir_edit.text()
        train_data_root_path = './buffer/train_data_root.txt'
        train_file = open(train_data_root_path, mode='a')
        train_file.write('\n'+fname)
        train_file.close()
        # initialize the loss figure
        self.resize_image(path="./buffer/blank.jpg", mode='loss')
        loss_im = QPixmap("./buffer/loss_buffer.jpg")
        loss_im = loss_im.scaledToWidth(430)
        loss_im = loss_im.scaledToHeight(300)
        self.loss_label.setPixmap(loss_im)

        """ begin to train the cnn model """
        self.statics_label.setText('begin to train, launching..., please waite...')
        self.train_sp = subprocess.Popen('bash ./buffer/train_cnn_model.sh', shell=True)
        pid = self.train_sp.pid
        print('Train cnn model pid: ', pid)

        self.show_log = True

        self.show_log_thread = threading.Thread(target=self.fill_log, name="fill_log_thread")
        self.show_log_thread.start()

    def stop_train_func(self):
        self.show_log = False
        os.remove('./buffer/log.log')
        os.remove('./buffer/train_data_root.txt')
        os.remove('./buffer/loss.jpg')
        os.remove('./buffer/loss_buffer.jpg')

        # pid = self.train_thread.pid
        # print(pid)
        os.kill(self.train_sp.pid + 3, signal.SIGTERM) # self.train_thread.pid + 3, because we don't know why the pid should pluse 3

    def resize_image(self, path, mode):
        img = Image.open(path)
        if mode == 'test':
            img = img.resize((430, 430))
            img.save("./buffer/test_buffer.jpg")
        elif mode == 'loss':
            img = img.resize((430, 300))
            img.save("./buffer/loss_buffer.jpg")

    def make_prediction(self):
        im_path = self.test_dir_edit.text()
        labels = {'neckline_design_labels': ['不存在', 'V领', '圆领', '深V领', '方领', '不规则领', '抹胸领', '一字领', '露肩领', '半开领', '桃形领'],
                  'collar_design_labels': ['不存在', '娃娃领', '清道夫领', '衬衫领', '飞行员领'],
                  'neck_design_labels': ['不存在', '荷叶半高领', '常规半高领', '堆堆领', '高常规领'],
                  'lapel_design_labels': ['不存在', '西装领', '一片领', '青果领', '直线领'],
                  'sleeve_length_labels': ['不存在', '无袖', '杯袖', '短袖', '五分袖', '七分袖', '九分袖', '长袖', '超长袖'],
                  'coat_length_labels': ['不存在', '高腰', '正常', '长款', '加长款', '及膝', '超长', '及地'],
                  'skirt_length_labels': ['不存在', '短袖', '中裙', '七分裙', '九分群', '长裙'],
                  'pant_length_labels': ['不存在', '短裤', '五分裤', '七分裤', '九分裤', '长裤']}

        tasks_all = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels', 'sleeve_length_labels',
                     'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 'pant_length_labels']

        results = inference(im_path)

        for idx, key,  in enumerate(tasks_all):
            a_predition = results[key]
            a_predition = labels[key][int(a_predition[0])] + '\n' + 'p = ' + str(a_predition[1])

            if key == tasks_all[0]:
                self.collar_result.setText(a_predition)
            elif key == tasks_all[1]:
                self.neckline_result.setText(a_predition)
            elif key == tasks_all[2]:
                self.skirt_result.setText(a_predition)
            elif key == tasks_all[3]:
                self.sleeve_result.setText(a_predition)
            elif key == tasks_all[4]:
                self.neck_result.setText(a_predition)
            elif key == tasks_all[5]:
                self.coat_result.setText(a_predition)
            elif key == tasks_all[6]:
                self.lapel_result.setText(a_predition)
            elif key == tasks_all[7]:
                self.pant_result.setText(a_predition)

        self.pe.setColor(QPalette.Background, Qt.lightGray)
        self.results_label.setPalette(self.pe)
        self.results_label.setText('Results')

    def while_testing(self):
        fname = self.test_dir_edit.text()
        # show test image
        self.resize_image(path=fname, mode='test')
        test_im = QPixmap("./buffer/test_buffer.jpg")
        test_im = test_im.scaledToWidth(430)
        test_im = test_im.scaledToHeight(430)
        self.image_label.setPixmap(test_im)

        # show other nformation
        self.pe.setColor(QPalette.Background, Qt.green)
        self.results_label.setPalette(self.pe)
        self.results_label.setText('testing... please waite')

        self.collar_result.setText('')
        self.neckline_result.setText('')
        self.skirt_result.setText('')
        self.sleeve_result.setText('')
        self.neck_result.setText('')
        self.coat_result.setText('')
        self.lapel_result.setText('')
        self.pant_result.setText('')



print('This GUI pid: ', os.getpid())
app = QApplication(sys.argv)
ex = ClassificationUI()
sys.exit(app.exec_())
