from cv2 import bitwise_and
import numpy as np
import cv2

#Canny function to try to use it in Function 2
'''
  * @desc opens a modal window to display a message
  * @param $image - image that apply canny on it
  * @return canny
'''

def canny(image):
    #1 convert to gray scale
        # Using cv2.cvtColor() method
        # Using cv2.COLOR_RGB2GRAY color space
        # conversion code
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #2 apply gaussian blur
        # Using cv2.GaussianBlur() method
        # Matrix Size of 5x5
        # conversion code
    blur=cv2.GaussianBlur(gray,(5,5),0)

    #3 apply the canny function to outline strong gradients
        # Image  : Input image to which Canny filter will be applied
        # T_lower: Lower threshold value in Hysteresis Thresholding
        # T_upper: Upper threshold value in Hysteresis Thresholding
    canny=cv2.Canny(blur,50,150)

    return canny



###Function 2: Process Binary Thresholded Images ###

def binary_thresholded(img):
    # Transform image to gray scale
        # Using cv2.cvtColor() method
        # Using cv2.COLOR_RGB2GRAY color space
        # conversion code
    gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
        # cv2.Sobel( (input) image. , An integer variable representing the depth of the imag,  x-derivative. , y-derivative )
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Scale result to 0-255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1
     
    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1


    # Convert image to HLS(HSL is pure white)
        # Using cv2.cvtColor() method
        # Using cv2.COLOR_BGR2HLS color space
        # conversion code
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Convert image to HSV (maximum value/brightness in HSV is analogous to shining a white light on a colored object)
        # Using cv2.cvtColor() method
        # Using cv2.COLOR_BGR2HSV color space
        # conversion code
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    H = hls[:,:,0]
    L=hls[:,:,1]
    S = hls[:,:,2]
    V=hsv[:,:,2]

    sat_binary = np.zeros_like(S)
    # Detect pixels that have a high saturation value
    sat_binary[(S > 90) & (S <= 255)] = 1

    hue_binary =  np.zeros_like(H)
    hue_binary[(H > 10)&(H<25)] = 1

    light_binary =  np.zeros_like(L)
    light_binary[(L>200)]=1

    v_binary=np.zeros_like(V)
    v_binary[(V>50)&(V<100)]=1


    # Try different combinations
    binary_1 = cv2.bitwise_or(sx_binary, white_binary)
    binary = cv2.bitwise_or(binary_1, sat_binary)
    binary2 = cv2.bitwise_or(binary, canny(img))
    
    return binary
