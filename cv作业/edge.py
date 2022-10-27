
import scipy
from PIL import Image, ImageFilter  # pillow package
import numpy as np
from scipy import ndimage



def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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



def gaussian(image,sigma):
    im = image
    #print(im.shape)
    #im_blur = np.zeros(im.shape,dtype=np.uint8)
    # for i in range(3): #caisetuxiang
    #     im_blur[:,:,i] = scipy.ndimage.gaussian_filter(im[:,:,i],sigma)
    im_blur = scipy.ndimage.gaussian_filter(im, sigma)
    return im_blur




def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    r, c = arr.shape
    G = np.zeros((r, c))
    #new_imageX = np.zeros(img.shape)
    #new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # for i in range(r - 2):
    #     for j in range(c - 2):
    #         new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
    #         new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
    new_imageX=ndimage.convolve(arr,s_suanziX)
    new_imageY=ndimage.convolve(arr,s_suanziY)
            #new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] **2+ new_imageY[i + 1, j + 1] **2) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    #G=np.uint8(new_image)
    Gx=new_imageX
    Gy=new_imageY
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            # x = G_x * arr[i - 1:i + 2, j - 1:j + 2]
            # Gx[i, j] = x.sum()
            # y = G_y * new
            # Gy[i, j] = y.sum()
            G[i, j] = np.sqrt(np.square((Gx[i, j])) + np.square((Gy[i, j])))
    #G=np.sqrt(np.square(Gx)+np.square(Gy))
    save_array_as_img(G,r'data/2.3_G.jpg')
    save_array_as_img(Gx,r'data/2.3_G_x.jpg')
    save_array_as_img(Gy, r'data/2.3_G_y.jpg')
    return G,Gx,Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here
    #G=rgb2gray(G)
    # Gy=rgb2gray(Gy)
    # Gx=rgb2gray(Gx)
    h,w=G.shape
    suppressed_G = np.zeros(G.shape)
    #theta = np.arctan2(Gy, Gx) * 180 / np.pi
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # if G[i, j] == 0:
            #     G[i, j] = 0
            # else:
                theta = np.arctan2(Gy[i, j], Gx[i, j]) * 180 / np.pi
                if (theta >= -22.5 and theta <= 22.5) or (theta >= 157.5 and theta <= 180) or (
                        theta >= -180 and theta <= -157.5):
                    N1 = G[i, j + 1]
                    N2 = G[i, j - 1]
                elif (theta >= 22.5 and theta <= 67.5) or (theta >= -157.5 and theta <= -112.5):
                    N1 = G[i + 1, j + 1]
                    N2 = G[i - 1, j - 1]
                elif (theta >= 67.5 and theta <= 112.5) or (theta >= -112.5 and theta <= -67.5):
                    N1 = G[i + 1, j]
                    N2 = G[i - 1, j]
                elif (theta >= 112.5 and theta <= 157.5) or (theta >= -67.5 and theta <= -22.5):
                    N1 = G[i - 1, j + 1]
                    N2 = G[i + 1, j - 1]

                if G[i][j]<N1 or G[i][j]<N2:
                    suppressed_G[i][j]=0
                else:
                    suppressed_G[i,j]=G[i,j]
                # G[i][j][G[i][j]<N1]=0
                # G[i][j][G[i][j]<N2]=0
    #suppressed_G=G
    save_array_as_img(suppressed_G,r'data/2.4_supress.jpg')
    return suppressed_G

def thresholding(G, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    #G=rgb2gray(G)
    # LT=low*np.max(G)
    # HT=high*np.max(G)
    LT=low
    HT=high
    h, w = G.shape
    Gl=G
    Gh=G
    G[G >= HT] = 255
    G[G <= LT] = 0
    Gl[Gl<= LT] = 0
    save_array_as_img(Gl, r'data/2.5_edgemap_low.jpg')
    Gh[Gh <= HT] = 0
    save_array_as_img(Gh, r'data/2.5_edgemap_high.jpg')
    nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)
    for i in range(1, h - 2):
        for j in range(1, w - 2):
            # G[G[i,j]<LT]=0
            # #Gl[Gl[i,j]<LT] = 0
            # G[G[i, j] > HT] = 255
            #Gl[Gh[i, j] > HT] = 255
            if G[i,j]>LT and G[i,j]<HT:
            # 把大于LT ，小于HT的点使用8连通区域确定
                if np.max(G[i - 1:i + 2, j - 1:j + 2] * nn) >= HT:
                    G[i, j] = 255
                else:
                    G[i, j] = 0

    save_array_as_img(G,r'data/2.5_edgemap.jpg')

    return G

def hough(G):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    pass

if __name__ == '__main__':
    input_path = 'data/road.jpeg'
    img = read_img_as_array(input_path)
    #show_array_as_img(img)
    #TODO: finish assignment Part II: detect edges on 'img'
#========================gray picture===========================
    imgrey =rgb2gray(img)
    save_array_as_img(imgrey,r'data/2.1_gray.jpg')
#========================gaussian================================
    blur=gaussian(imgrey,3)
    save_array_as_img(blur,r'data/2.2_gauss.jpg')

#=====================sober====================================
    #img2=Image.open(r'data/2.1_gray.jpg')
    G0=sobel(blur)
#=====================nonmax-supress=============================
    nomax=nonmax_suppress(G0[0],G0[1],G0[2])
#====================threshold===============================
    thresholding(nomax,20,110)

