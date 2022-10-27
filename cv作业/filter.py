
from PIL import Image # pillow package
import numpy as np
from scipy import ndimage
import copy

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################
def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # TODO: Please complete this function.
    # your code here
    im=read_img_as_array(img)
    im_blur = np.zeros(im.shape, dtype=np.uint8)
    im_blur = ndimage.gaussian_filter(im, sigma)
    # for i in range(3):
    #     im_blur[:,:,i] = ndimage.gaussian_filter(im[:,:,i],sigma)
    detail = im - im_blur
    sharpenimg=im+alpha*detail
    sharpenimg[sharpenimg<0]=0
    sharpenimg[sharpenimg>255]=255
    show_array_as_img(sharpenimg)
    #print(sharpenimg)
    save_array_as_img(sharpenimg,r'data/1.1_sharpened.jpg')

def median_filter(img, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    # your code here
    img=read_img_as_array(img)
    im=img
    print(im)
    R = im[:, :, 0]  # red channel
    G = im[:, :, 1]  # green channel
    B = im[:, :, 2]  # blue channel

    height, width = R.shape
    new_array_red = np.zeros((height, width), dtype=int)
    new_array_green = np.zeros((height, width), dtype=int)
    new_array_blue = np.zeros((height, width), dtype=int)

    mid = int((s - 1) / 2)
    for i in range(mid, height - mid):
        for j in range(mid, width - mid):
            new_array_red[i, j] = np.median(
                R[i - mid:i + mid + 1, j - mid:j + mid + 1])
            new_array_green[i, j] = np.median(
                G[i - mid:i + mid + 1, j - mid:j + mid + 1])
            new_array_blue[i, j] = np.median(
                B[i - mid:i + mid + 1, j - mid:j + mid + 1])
    new_array = np.array([new_array_red, new_array_green, new_array_blue])
    new_array = np.transpose(new_array, (1, 2, 0))
    print('new_array', new_array.shape)
    img2 = Image.fromarray(np.uint8(new_array))
    print("结束")
    #print(img2)
    show_array_as_img(new_array)
    save_array_as_img(new_array,r'data/1.2_derained.jpg')



if __name__ == '__main__':
    input_path = 'data/rain.jpeg'
    # img = read_img_as_array(input_path)
    # show_array_as_img(img)
    sharpen(input_path,3,2)
    median_filter(input_path,5)
    #TODO: finish assignment Part I.
