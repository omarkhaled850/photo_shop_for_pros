## photo_shop_for_pros
this is a python PC program for digital image processing 

Name : Omar Khaled Mohammed 

program details:
	language                          : python (3.8)
	image processing framwork 		  : opencv (cv2)
	graphical user interface framwork : QT5
	
what the toolbox do :

***Image Operations***
gray scale
Image flip(around x, around y, around x&y)
Image rotation (at any degree)
Image skewing
Image translation
Image crop
Scaling (zoom in or zoom out)

***Image Enhancement***
***1) Point Processing***
-Brightness Increase(Log transformation Power(gama) transformation)

-Histogram equalization
-Brightness decrease(Log transformation Power(gama) transformation)
-Image negative
-bit slicing
-gray level slicing

**2) Neighborhood operations Processing**
-Smoothing(traditional 3*3 filter, pyramidal filter, circular filter, cone filter, median blur)
-Sharpening(sobel)

***Image Segmentation***
Image Segmentation





How to run the program: 
	
	1- I have made an exe package ready to run without installing any thing
		-in the toolbox file open "exe file" folder then "dist" run "main.exe"
	2- If the exe did not follow the step to setup the required software
		steps:
			1- you will find the python installer in the folder "installation" install it and make sure the you add python to PATH
			2- a restart may be required
			3- open the command prompt in the toolbox file and copy and past the follewing line 
			   "pip install opencv-python PyQt5 imutils pyqt5-tools" you may need to type "y" and click inter to confirm
			4- after the installation type "python main.py" to RUN the program




