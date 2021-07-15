
# coding: utf-8

# # BNN on Pynq
# 
# This notebook covers how to use Binary Neural Networks on Pynq. 
# It shows an example of handwritten digit recognition using a binarized neural network composed of 4 fully connected layers with 1024 neurons each, trained on the MNIST dataset of handwritten digits. 
# In order to reproduce this notebook, you will need an external USB Camera connected to the PYNQ Board.
# 
# ## 1. Import the package

# In[1]:


import bnn


# ## 2. Checking available parameters
# 
# By default the following trained parameters are available for LFC network using 1 bit for weights and 1 threshold for activation:

# In[2]:


print(bnn.available_params(bnn.NETWORK_LFCW1A2))


# Two sets of weights are available for the LFCW1A1 network, the MNIST and one for character recognition (NIST).

# ## 3. Instantiate the classifier
# 
# Creating a classifier will automatically download the correct bitstream onto the device and load the weights trained on the specified dataset. This example works with the LFCW1A1 for inferring MNIST handwritten digits.
# Passing a runtime attribute will allow to choose between hardware accelerated or pure software inference.

# In[3]:


hw_classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A2,"5class_LBP_FER",bnn.RUNTIME_HW)
sw_classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A2,"5class_LBP_FER",bnn.RUNTIME_SW)


# In[4]:


print(hw_classifier.classes)


# ## 4. Load the image from the camera
# The image is captured from the external USB camera and stored locally. The image is then enhanced in contract and brightness to remove background noise. 
# The resulting image should show the digit on a white background:

# In[16]:


import cv2
from PIL import Image as PIL_Image
from PIL import ImageEnhance
from PIL import ImageOps

# says we capture an image from a webcam
cap = cv2.VideoCapture(0) 
_ , cv2_im = cap.read()
cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
img = PIL_Image.fromarray(cv2_im).convert("L") 

#original captured image
#orig_img_path = '/home/xilinx/jupyter_notebooks/bnn/pictures/webcam_image_mnist.jpg'
#img = PIL_Image.open(orig_img_path).convert("L")     
                   
#Image enhancement                
contr = ImageEnhance.Contrast(img)
img = contr.enhance(1)  #ajaybs#3                                                  # The enhancement values (contrast and brightness) 
bright = ImageEnhance.Brightness(img)                                     # depends on backgroud, external lights etc
img = bright.enhance(3.0)   #jaybs#4.0       

#img = img.rotate(180)                                                     # Rotate the image (depending on camera orientation)
#Adding a border for future cropping
img = ImageOps.expand(img,border=80,fill='white') 
img


# In[17]:


from PIL import Image as PIL_Image
import numpy as np
import math
from scipy import misc

#Find bounding box  
inverted = ImageOps.invert(img)  
box = inverted.getbbox()  
img_new = img.crop(box)  
width, height = img_new.size  
ratio = min((28./height), (28./width))  
background = PIL_Image.new('RGB', (28,28), (255,255,255))  
if(height == width):  
    img_new = img_new.resize((28,28))  
elif(height>width):  
    img_new = img_new.resize((int(width*ratio),28))  
    background.paste(img_new, (int((28-img_new.size[0])/2),int((28-img_new.size[1])/2)))  
else:  
    img_new = img_new.resize((28, int(height*ratio)))  
    background.paste(img_new, (int((28-img_new.size[0])/2),int((28-img_new.size[1])/2)))  
  
background  
img_data=np.asarray(background)  
img_data = img_data[:,:,0]  
misc.imsave('/home/xilinx/jupyter_notebooks/img_webcam_mnist1.png', img_data)


# In[18]:


# VIOLA JONES ALGORITHM

import cv2
import sys
import os
import glob
from scipy import misc

misc.imsave('/home/xilinx/jupyter_notebooks/img_webcam_mnist1.png', img)

CASCADE_PATH = "/home/xilinx/jupyter_notebooks/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier("/home/xilinx/jupyter_notebooks/haarcascade_frontalface_default.xml")
img = cv2.imread('/home/xilinx/jupyter_notebooks/img_webcam_mnist1.png')
#ajaybs#img = cv2.imread('/home/xilinx/jupyter_notebooks/test_HAPPY_image3.jpg')
if (img is None):
    print("Can't open image file")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(img, 1.3, 5)
if (faces is None):
    print('Failed to detect face')

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

facecnt = len(faces)
print("Detected faces: %d" % facecnt)
height, width = img.shape[:2]

for (x, y, w, h) in faces:
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)

    faceimg = img[ny:ny+nr, nx:nx+nr]
    lastimg = cv2.resize(faceimg, (28, 28))
    
    print("VIOLA_JONES_FACE_CROPPED_IMAGE")
    misc.imsave('/home/xilinx/img_webcam_VJ.jpg', lastimg)

img_load = PIL_Image.open('/home/xilinx/img_webcam_VJ.jpg').convert("L")  
img_load


# ## 5. Crop and scale the image
# The center of mass of the image is evaluated to properly crop the image and extract the written digit only. 

# In[11]:


#ajaybs#from PIL import Image as PIL_Image
#ajaybs#import numpy as np
#ajaybs#import math
#ajaybs#from scipy import misc

#ajaybs##Find bounding box  
#ajaybs#inverted = ImageOps.invert(img)  
#ajaybs#box = inverted.getbbox()  
#ajaybs#img_new = img.crop(box)  
#ajaybs#width, height = img_new.size  
#ajaybs#ratio = min((28./height), (28./width))  
#ajaybs#background = PIL_Image.new('RGB', (28,28), (255,255,255))  
#ajaybs#if(height == width):  
    #ajaybs#img_new = img_new.resize((28,28))  
#ajaybs#elif(height>width):  
    #ajaybs#img_new = img_new.resize((int(width*ratio),28))  
    #ajaybs#background.paste(img_new, (int((28-img_new.size[0])/2),int((28-img_new.size[1])/2)))  
#ajaybs#else:  
    #ajaybs#img_new = img_new.resize((28, int(height*ratio)))  
    #ajaybs#background.paste(img_new, (int((28-img_new.size[0])/2),int((28-img_new.size[1])/2)))  
  
#ajaybs#background  
#ajaybs#img_data=np.asarray(background)  
#ajaybs#img_data = img_data[:,:,0]  
#ajaybs#misc.imsave('/home/xilinx/img_webcam_mnist.jpg', img_data)

 


# In[12]:


# code for LBP - start
import numpy as np
import cv2
from matplotlib import pyplot as plt

def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return [0,0,0] 

#ajaybs#img = cv2.imread('fer0000001.png', 0)
#ajaybs8Apr#img_load = PIL_Image.open('/home/xilinx/img_webcam_VJ.jpg').convert("L")  
img_load = PIL_Image.open('/home/xilinx/img_webcam_VJ.jpg').convert("L")  

img_load
transformed_img = cv2.imread('/home/xilinx/img_webcam_VJ.jpg', 0)
transformed_img
#ajaybs#filelist = ['/home/ajay/Documents/FER2013_create_png/FER2013Train_HAPPY/fer0000001.png','/home/ajay/Documents/FER2013_create_png/FER2013Train_HAPPY/fer0000002.png','/home/ajay/Documents/FER2013_create_png/FER2013Train_HAPPY//fer0000003.png','/home/ajay/Documents/FER2013_create_png/FER2013Train_HAPPY/fer0000004.png']
cnt = 0
#for imagepath in path:
img = cv2.imread('/home/xilinx/img_webcam_VJ.jpg', 0)
for x in range(0, 27):
    for y in range(0, 27):
        values = []
        center        = img[x,y]
        top_left      = get_pixel_else_0(img, x-1, y-1)
        top_up        = get_pixel_else_0(img, x, y-1)
        top_right     = get_pixel_else_0(img, x+1, y-1)
        right         = get_pixel_else_0(img, x+1, y )
        left          = get_pixel_else_0(img, x-1, y )
        bottom_left   = get_pixel_else_0(img, x-1, y+1)
        bottom_right  = get_pixel_else_0(img, x+1, y+1)
        bottom_down   = get_pixel_else_0(img, x,   y+1 )

        #ajaybs#print bottom_right
        #ajaybs#print x
        #ajaybs#print y
        #ajaybs#values = thresholded(center, [top_left[0], top_up[0], top_right[0], right[0], bottom_right[0],
                                      #ajaybs#bottom_down[0], bottom_left[0], left[0]])

        if top_left >= center:
            values.append(1)
        else:
            values.append(0)

        if top_up >= center:
            values.append(1)
        else:
            values.append(0)


        if top_right >= center:
            values.append(1)
        else:
            values.append(0)

        if right >= center:
            values.append(1)
        else:
            values.append(0)


        if bottom_right >= center:
            values.append(1)
        else:
            values.append(0)


        if bottom_down >= center:
            values.append(1)
        else:
            values.append(0)


        if bottom_left >= center:
            values.append(1)
        else:
            values.append(0)


        if left >= center:
            values.append(1)
        else:
            values.append(0)




        weights = [1, 2, 4, 8, 16, 32, 64, 128]
        res = 0
        for a in range(0, len(values)):
            res += weights[a] * values[a]

        transformed_img.itemset((x,y), res)

#misc.imsave('smallimg', transformed_img)

print("transformed_img")
misc.imsave('/home/xilinx/img_webcam_VJ_LBP.jpg', transformed_img)

img_load = PIL_Image.open('/home/xilinx/img_webcam_VJ_LBP.jpg').convert("L")  
img_load

#code for LBP - end


# ## 6. Convert to BNN input format
# The image is resized to comply with the MNIST standard. The image is resized at 28x28 pixels and the colors inverted. 

# In[27]:


from array import *
from PIL import Image as PIL_Image
from PIL import ImageOps
img_load = PIL_Image.open('/home/xilinx/img_webcam_VJ_LBP.jpg').convert("L")  
# Convert to BNN input format  
# The image is resized to comply with the MNIST standard. The image is resized at 28x28 pixels and the colors inverted.   
  
#Resize the image and invert it (white on black)  
smallimg = img_load
smallimg = smallimg.rotate(0)  
#ajaybs#print ("smallimg before\n") #ajaybs#
#ajaybs#smallimg   #ajaybs#
data_image = array('B')  
  
pixel = smallimg.load()  
for x in range(0,28):  
    for y in range(0,28):  
        if(pixel[y,x] == 255):  
            data_image.append(255)  
        else:  
            data_image.append(1)  
# Setting up the header of the MNIST format file - Required as the hardware is designed for MNIST dataset         
hexval = "{0:#0{1}x}".format(1,6)  
header = array('B')  
header.extend([0,0,8,1,0,0])  
header.append(int('0x'+hexval[2:][:2],16))  
header.append(int('0x'+hexval[2:][2:],16))  
header.extend([0,0,0,28,0,0,0,28])  
header[3] = 3 # Changing MSB for image data (0x00000803)  
data_image = header + data_image  
output_file = open('/home/xilinx/img_webcam_mnist_processed', 'wb')  
data_image.tofile(output_file)  
output_file.close()   
smallimg

print ("data_image\n")
data_image
print ("smallimg after\n")
smallimg



# ## 7. Launching BNN in hardware
# 
# The image is passed in the PL and the inference is performed. Use `classify_mnist` to classify a single mnist formatted picture.

# In[28]:


class_out = hw_classifier.classify_mnist("/home/xilinx/img_webcam_mnist_processed")
print("Class number: {0}".format(class_out))
print("Class name: {0}".format(hw_classifier.class_name(class_out)))


# ## 8. Launching BNN in software
# The inference on the same image is performed in sofware on the ARM core

# In[16]:


class_out=sw_classifier.classify_mnist("/home/xilinx/img_webcam_mnist_processed")
print("Class number: {0}".format(class_out))
print("Class name: {0}".format(hw_classifier.class_name(class_out)))


# ## 9. Reset the device

# In[16]:


from pynq import Xlnk

xlnk = Xlnk()
xlnk.xlnk_reset()


# In[ ]:




