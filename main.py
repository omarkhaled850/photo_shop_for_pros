# Name : Omar Khaled Mohammed 


import imutils
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QAction, QFileDialog, QMainWindow, QDialog, QDialogButtonBox, QVBoxLayout, QInputDialog, \
    QLineEdit


# dialog message
class CustomDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("HELLO!")

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

# qweight
class Ui_MainWindow(QMainWindow):
    global_img = None
    display_img = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(0, 0, 161, 421))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.main_toolbox = QtWidgets.QToolBox(self.formLayoutWidget)
        self.main_toolbox.setEnabled(True)
        self.main_toolbox.setMouseTracking(False)
        self.main_toolbox.setObjectName("main_toolbox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 159, 338))
        self.page.setObjectName("page")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.page)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, -1, 160, 341))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gray_scale_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.gray_scale_btn.setObjectName("gray_scale_btn")
        self.verticalLayout.addWidget(self.gray_scale_btn)
        self.flip_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.flip_btn.setObjectName("flip_btn")
        self.verticalLayout.addWidget(self.flip_btn)
        self.rot_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.rot_btn.setObjectName("rot_btn")
        self.verticalLayout.addWidget(self.rot_btn)
        self.skewing_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.skewing_btn.setObjectName("skewing_btn")
        self.verticalLayout.addWidget(self.skewing_btn)

        self.croping_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.croping_btn.setObjectName("croping_btn")
        self.verticalLayout.addWidget(self.croping_btn)

        self.scaling_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.scaling_btn.setObjectName("scaling_btn")
        self.verticalLayout.addWidget(self.scaling_btn)
        self.translation_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.translation_btn.setObjectName("translation_btn")
        self.verticalLayout.addWidget(self.translation_btn)

        self.main_toolbox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 159, 338))
        self.page_2.setObjectName("page_2")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.page_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(-20, 0, 187, 295))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.negative_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.negative_btn.setObjectName("negative_btn")
        self.verticalLayout_2.addWidget(self.negative_btn)

        self.hist_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.hist_btn.setObjectName("hist_btn")
        self.verticalLayout_2.addWidget(self.hist_btn)

        self.log_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.log_btn.setObjectName("log_btn")
        self.verticalLayout_2.addWidget(self.log_btn)
        self.gamma_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.gamma_btn.setObjectName("gamma_btn")
        self.verticalLayout_2.addWidget(self.gamma_btn)

        self.blending_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.blending_btn.setObjectName("blending_btn")
        self.verticalLayout_2.addWidget(self.blending_btn)

        self.bitslicing_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.bitslicing_btn.setObjectName("bitslicing_btn")
        self.verticalLayout_2.addWidget(self.bitslicing_btn)

        self.slicing_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.slicing_btn.setObjectName("slicing_btn")
        self.verticalLayout_2.addWidget(self.slicing_btn)




        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.smoothing_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.smoothing_btn.setObjectName("smoothing_btn")
        self.verticalLayout_2.addWidget(self.smoothing_btn)
        self.sharp_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.sharp_btn.setObjectName("sharp_btn")
        self.verticalLayout_2.addWidget(self.sharp_btn)

        self.main_toolbox.addItem(self.page_2, "")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setGeometry(QtCore.QRect(0, 0, 159, 338))
        self.page_3.setObjectName("page_3")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.page_3)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(-10, 0, 171, 80))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.threshold_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.threshold_btn.setObjectName("threshold_btn")
        self.verticalLayout_3.addWidget(self.threshold_btn)
        self.edge_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.edge_btn.setObjectName("edge_btn")
        self.verticalLayout_3.addWidget(self.edge_btn)
        self.main_toolbox.addItem(self.page_3, "")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.main_toolbox)
        self.image_out = QtWidgets.QLabel(self.centralwidget)
        self.image_out.setGeometry(QtCore.QRect(180, 0, 581, 441))
        self.image_out.setText("")
        ##############################################################################################
        self.image_out.setPixmap(QtGui.QPixmap(""))
        self.image_out.setScaledContents(True)
        self.image_out.setObjectName("image_out")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFiles = QtWidgets.QMenu(self.menubar)
        self.menuFiles.setObjectName("menuFiles")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpenFile = QtWidgets.QAction(MainWindow)
        self.actionOpenFile.setObjectName("actionOpenFile")
        self.menuFiles.addAction(self.actionOpenFile)
        self.menubar.addAction(self.menuFiles.menuAction())

        self.retranslateUi(MainWindow)
        self.main_toolbox.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.rot_btn.clicked.connect(self.rotate_fun)

    def retranslateUi(self, MainWindow):

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.gray_scale_btn.setText(_translate("MainWindow", "gray image"))
        self.flip_btn.setText(_translate("MainWindow", "flip image"))
        self.rot_btn.setText(_translate("MainWindow", "rotate image"))
        self.skewing_btn.setText(_translate("MainWindow", "skewing"))
        self.croping_btn.setText(_translate("MainWindow", "crop"))
        self.scaling_btn.setText(_translate("MainWindow", "scaling"))
        self.translation_btn.setText(_translate("MainWindow", "image translation"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page),
                                      _translate("MainWindow", "Image Operations"))
        self.label.setText(_translate("MainWindow", "        1-Point Processing"))
        self.hist_btn.setText(_translate("MainWindow", "histogram equalization"))
        self.log_btn.setText(_translate("MainWindow", "Log transformation"))
        self.gamma_btn.setText(_translate("MainWindow", "Power transformation"))
        self.blending_btn.setText(_translate("MainWindow", "Image blending"))
        self.bitslicing_btn.setText(_translate("MainWindow", "bite plane slicing"))
        self.slicing_btn.setText(_translate("MainWindow", "Image slicing"))
        self.negative_btn.setText(_translate("MainWindow", "negative"))

        self.label_2.setText(_translate("MainWindow", "         2-Neighborhood Processing"))
        self.smoothing_btn.setText(_translate("MainWindow", "smoothing"))
        self.sharp_btn.setText(_translate("MainWindow", "sharping"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page_2),
                                      _translate("MainWindow", "Image Enhancement"))
        self.threshold_btn.setText(_translate("MainWindow", "thresholding"))
        self.edge_btn.setText(_translate("MainWindow", "edge segmantation"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page_3),
                                      _translate("MainWindow", "Image Enhancement"))
        self.threshold_btn.setText(_translate("MainWindow", "thresholding"))
        self.edge_btn.setText(_translate("MainWindow", "edge segmantation"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page_3),
                                      _translate("MainWindow", "Image Segmentation"))
        self.menuFiles.setTitle(_translate("MainWindow", "Files"))
        self.actionOpenFile.setText(_translate("MainWindow", "OpenFile"))
        # connecting the open image function with the menbar button
        self.actionOpenFile.triggered.connect(self.open_img)

        # connecting buttons with functions
        self.gray_scale_btn.clicked.connect(self.graScaleFun)
        self.flip_btn.clicked.connect(self.flip_fun)
        self.skewing_btn.clicked.connect(self.skewing_fun)
        self.croping_btn.clicked.connect(self.crop_fun)
        self.scaling_btn.clicked.connect(self.scaling_fun)
        self.translation_btn.clicked.connect(self.trans_fun)
        self.hist_btn.clicked.connect(self.histequ_fun)
        self.log_btn.clicked.connect(self.log_fun)
        self.gamma_btn.clicked.connect(self.gammacorrection)
        self.blending_btn.clicked.connect(self.blend_fun)
        self.slicing_btn.clicked.connect(self.slicing_fun)
        self.negative_btn.clicked.connect(self.negative_fun)
        self.smoothing_btn.clicked.connect(self.smoothing_fun)
        self.sharp_btn.clicked.connect(self.sharp_fun)
        self.threshold_btn.clicked.connect(self.threshold_fun)
        self.edge_btn.clicked.connect(self.edge_fun)
        self.bitslicing_btn.clicked.connect(self.bitslicing_fun)


#open a dialog fo the user to choose the image


    def open_img(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        self.global_img = cv2.imread(imagePath)
        self.display_img = QPixmap(imagePath)
        self.image_out.setPixmap(self.display_img)
        self.resize(self.display_img.size())
        self.adjustSize()
        self.image_data()
    

    def bitslicing_fun(self):
        lst = []
        for i in range(self.global_img.shape[0]):
            for j in range(self.global_img.shape[1]):
                lst.append(np.binary_repr(self.global_img[i][j], width=8))
        eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(self.global_img.shape[0],self.global_img.shape[1])
        self.global_img = eight_bit_img
        self.displayINlebal()


    def OpenTextFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setWindowTitle("Choose a file to open")
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        dialog.setNameFilter("Text (*.txt);; All files (*.*)")
        dialog.setViewMode(QtGui.QFileDialog.Detail)

        filename = QtCore.QStringList()

        if (dialog.exec_()):
            file_name = dialog.selectedFiles()
        plain_text = open(file_name[0]).read()
        self.Editor.setPlainText(plain_text)
        self.file_path = str(file_name[0])

    def graScaleFun(self):
        gray_img = cv2.cvtColor(self.global_img, cv2.COLOR_BGR2GRAY)
        self.global_img = gray_img
        cv2.imwrite('savedImage.jpg', gray_img)
        self.image_out.setPixmap(QtGui.QPixmap("savedImage.jpg"))
        self.image_out.setScaledContents(True)

    def negative_fun(self):
        height, width = self.global_img.shape


        self.global_img= 255 - self.global_img

        negative_img = self.global_img
        # Display the negative transformed image
        self.displayINlebal()



    def getTextForFlip(self):
        text, okPressed = QInputDialog.getText(self, "fliping", "type:x or y or xy", QLineEdit.Normal, "")
        if okPressed and text != '':
            print(text)
            return text
    def flip_fun(self):
        res = self.getTextForFlip()
        if res == 'x':
            flipedimage = cv2.flip(self.global_img, 0)
        elif res == 'y':
            flipedimage = cv2.flip(self.global_img, 1)
        elif res == 'xy':
            flipedimage = cv2.flip(self.global_img, -1)

        self.global_img = flipedimage
        self.displayINlebal()
        return flipedimage

    def getAngle(self):
        angle, okPressed = QInputDialog.getDouble(self, "Get double", "Value:", 10.05, -360, 360, 10)
        if okPressed:
            print(angle)
        return angle

    def rotate_fun(self):
        angle = self.getAngle()
        rotated = imutils.rotate_bound(self.global_img, angle)
        self.global_img = rotated
        self.displayINlebal()

    def skewing_fun(self):

        a1, okPressed = QInputDialog.getDouble(self, "enter skewing destnation", "x value 1st point:", 0.0, -self.global_img.shape[0], self.global_img.shape[0], 1)
        a2, okPressed = QInputDialog.getDouble(self, "enter skewing destnation", "y value 1st point:", 0.0, -self.global_img.shape[1], self.global_img.shape[1], 1)
        b1, okPressed = QInputDialog.getDouble(self, "enter skewing destnation", "x value 2nd point:", 0.0, -self.global_img.shape[0], self.global_img.shape[0], 1)
        #b2, okPressed = QInputDialog.getDouble(self, "enter skewing destnation", "y value 2nd point:", 0.0, -self.global_img.shape[1], self.global_img.shape[1], 1)
        c1, okPressed = QInputDialog.getDouble(self, "enter skewing destnation", "x value 3rd point:", 0.0, -self.global_img.shape[0], self.global_img.shape[0], 1)
        #c2, okPressed = QInputDialog.getDouble(self, "enter skewing destnation", "y value 3rd point:", 0.0, -self.global_img.shape[1], self.global_img.shape[1], 1)
        image = self.global_img
        src_pts = np.float32([[0, 0], [image.shape[0] - 1, 0], [0, image.shape[1] - 1]])
        dst_pts = np.float32([[a1, a2], [image.shape[0]+b1, 0], [c1, image.shape[1]]])
        Mat = cv2.getAffineTransform(src_pts, dst_pts)
        skewed = cv2.warpAffine(image, Mat, (image.shape[1], image.shape[0]))
        self.global_img = skewed
        self.displayINlebal()

    def crop_fun(self):
        a1, okPressed = QInputDialog.getInt(self, "enter croping points", "from row number:", 0, -self.global_img.shape[0], self.global_img.shape[0], 1)
        a2, okPressed = QInputDialog.getInt(self, "enter croping points", "to row number:", 0, -self.global_img.shape[1], self.global_img.shape[1], 1)
        b1, okPressed = QInputDialog.getInt(self, "enter croping points", "from col number:", 0, -self.global_img.shape[0], self.global_img.shape[0], 1)
        b2, okPressed = QInputDialog.getInt(self, "enter croping points", "to col number:", 0, -self.global_img.shape[1], self.global_img.shape[1], 1)
        self.global_img = self.global_img[a1:a2, b1:b2]
        self.displayINlebal()
        cv2.imshow("Cropped Image", self.global_img)
        cv2.waitKey(0)

    def scaling_fun(self):
        scaleX, okPressed = QInputDialog.getInt(self, "Get scaling value", "fx Value:", 1, -self.global_img.shape[0], self.global_img.shape[0], 1)
        scaleY, okPressed = QInputDialog.getInt(self, "Get scaling value", "fy Value:", 1, -self.global_img.shape[1], self.global_img.shape[1], 1)
        img = self.global_img
        # Reduce the image to 0.6 times the original
        scaled_img = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        self.global_img = scaled_img
        self.displayINlebal()
        cv2.imshow("Scaled Image", self.global_img)
        cv2.waitKey(0)


    def histequ_fun(self):
        equ = cv2.equalizeHist(self.global_img)
        self.global_img = equ
        self.displayINlebal()



    def trans_fun(self):
        # Store height and width of the image
        gray_img = self.global_img
        height, width = gray_img.shape[:2]

        transX, okPressed = QInputDialog.getInt(self, "Get translation value", "x Value:", 1, -self.global_img.shape[0],self.global_img.shape[0], 1)
        transY, okPressed = QInputDialog.getInt(self, "Get translation value", "y Value:", 1, -self.global_img.shape[1],self.global_img.shape[1], 1)


        T = np.float32([[1, 0, transX], [0, 1, transY]])

        # We use warpAffine to transform
        # the image using the matrix, T
        trans_img = cv2.warpAffine(gray_img, T, (width, height))
        self.global_img = trans_img
        self.displayINlebal()

    def blend_fun(self):


        factor1, okPressed = QInputDialog.getDouble(self, "1st image factor:", 0.5, 0.0, 1, 0.1)
        factor2, okPressed = QInputDialog.getDouble(self, "2nd image factor:", 0.5, 0.0, 1, 0.1)

        flipedimage = cv2.flip(self.global_img, 0)
        dst = cv2.addWeighted(flipedimage, factor1, self.global_img, factor2, 0)
        self.global_img = dst
        self.displayINlebal()

    # # Apply log transformation method
    def log_fun(self):
        c = 255 / np.log(1 + np.max(self.global_img))
        log_image = c * (np.log(self.global_img + 1))
        log_image = np.array(log_image, dtype=np.uint8)
        self.global_img = log_image
        self.displayINlebal()

    # # gamma correction

    def gammacorrection(self):
        # Apply gamma correction.
        gamma, okPressed = QInputDialog.getDouble(self, "enter gamma value ", "gamma<0 = darker , gamma>0 = lighter ", 0.5, 0, 4, 3)

        gamma_corrected = np.array(255 * (self.global_img / 255) ** gamma, dtype='uint8')
        self.global_img = gamma_corrected
        self.displayINlebal()

    def threshold_fun(self):

        threshvalue, okPressed = QInputDialog.getDouble(self, "thresholding ", "enter the threshold value:",1, 0, 255, 1)
        ret, thresh1 = cv2.threshold(self.global_img, threshvalue, 255, cv2.THRESH_BINARY)
        self.global_img = thresh1
        self.displayINlebal()


        # grey level slicing

    def slicing_fun(self):
        # Find width and height of image
        row, column = self.global_img.shape
        # Create an zeros array to store the sliced image
        img1 = np.zeros((row, column), dtype='uint8')
        # Specify the min and max range
        min_range, okPressed = QInputDialog.getInt(self, "slicing data", "min range", 0, 0, 255, 1)
        max_range, okPressed = QInputDialog.getInt(self, "slicing data", "max range", 0, 0, 255, 1)

        # Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
        for i in range(row):
            for j in range(column):
                if self.global_img[i, j] > max_range:
                    img1[i, j] = 255
                elif self.global_img[i, j] < min_range:
                    img1[i, j] = 0

        # Display the image
        self.global_img = img1
        self.displayINlebal()

# this function takes the user choice and give him the filter he\she wants

    def smoothing_fun(self):
        items = ("Gaussian", "Averaging", "circular","pyramidal","cone","median")
        item, okPressed = QInputDialog.getItem(self, "choose method", "filter:", items, 0, False)
        if okPressed and item:
            print(item)
        if item == 'Gaussian':
            print(item)
        if item == 'Gaussian':
            self.global_img = cv2.GaussianBlur(self.global_img, (5, 5), 0)
            self.displayINlebal()
        elif item == "Averaging":
            self.global_img = cv2.blur(self.global_img, (5, 5))
            self.displayINlebal()
        elif item == "circular":
            circularKernal = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],np.float32) / 21
            dst = cv2.filter2D(self.global_img, -1, circularKernal)
            self.global_img = dst
            self.displayINlebal()
        elif item == "pyramidal":
            pykernel = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]],np.float32) / 81
            dst = cv2.filter2D(self.global_img, -1, pykernel)
            self.global_img = dst
            self.displayINlebal()
        elif item == "cone":
            coneKernel = np.array([[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1], [0, 2, 2, 2, 0], [0, 0, 1, 0, 0]],np.float32) / 25
            dst = cv2.filter2D(self.global_img, -1, coneKernel)
            self.global_img = dst
            self.displayINlebal()
        elif item == "median":
            median = cv2.medianBlur(self.global_img, 5)
            self.global_img = median
            self.displayINlebal()

    def sharp_fun(self):
        items = ("laplacian", "sobelx", "sobely",)
        item, okPressed = QInputDialog.getItem(self, "choose method", "filter:", items, 0, False)
        if item == "sobely":
            sobely = cv2.Sobel(self.global_img, cv2.CV_64F, 0, 1, ksize=5)
            self.global_img = sobely
        elif item == "sobelx":
            sobelx = cv2.Sobel(self.global_img, cv2.CV_64F, 1, 0, ksize=5)
            self.global_img = sobelx
        elif item == "laplacian":
            laplacian = cv2.Laplacian(self.global_img, cv2.CV_64F)
            self.global_img = laplacian
        self.displayINlebal()

    def edge_fun(self):
        # Remove noise by blurring with a Gaussian filter
        blur = cv2.GaussianBlur(self.global_img, (5, 5), 0)
        # Apply Laplace function
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        # converting back to uint8
        abs_dst = cv2.convertScaleAbs(laplacian)
        self.global_img = abs_dst
        self.displayINlebal()

# this function prints the data of any input image in console

    def image_data(self):
        print(type(self.global_img))
        print('RGB shape: ')  # Rows, cols, channels
        print(self.global_img.shape)
        imgshape = self.global_img.shape
        rows = imgshape[0]
        cols = imgshape[1]
        print(rows)
        print(cols)
        print('Gray shape:')
        print(self.global_img.shape)
        print('img.dtype: ')
        print(self.global_img.dtype)
        print('img.size: ')
        print(self.global_img.dtype)

# this function is for displaying the image in the fram of the program not in a imshow new fram


    def displayINlebal(self):
        filename = 'savedImage.jpg'
        cv2.imwrite(filename, self.global_img)
        self.image_out.setPixmap(QtGui.QPixmap("savedImage.jpg"))
        self.image_out.setScaledContents(True)
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
