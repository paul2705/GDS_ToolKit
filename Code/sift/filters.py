import os

import numpy as np
import math
import scipy 
import matplotlib.pyplot as plt
# import cv2

from common1 import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = []
    for i in range(math.ceil(image.shape[0]/patch_size[0])):
        for j in range(math.ceil(image.shape[1]/patch_size[1])):
            tmp = image[patch_size[0]*i:patch_size[0]*(i+1),patch_size[1]*j:patch_size[1]*(j+1)];
            avg = np.sum(np.sum(tmp))/tmp.shape[0]/tmp.shape[1];
            tmp = (tmp-avg)/np.sqrt(np.sum(np.sum((tmp-avg)**2))/tmp.shape[0]/tmp.shape[1]);
            #print(tmp.shape);
            output.append(tmp);
    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    output    = np.zeros(image.shape);
    flpKernel = kernel[::-1,::-1];
    print(flpKernel)
    h_2       = math.floor(kernel.shape[0]/2);
    w_2       = math.floor(kernel.shape[1]/2);
    padImage  = np.zeros((image.shape[0]+2*h_2,image.shape[1]+2*w_2));
    padImage[h_2:h_2+image.shape[0],w_2:w_2+image.shape[1]] = image;
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            tmp = padImage[i:i+flpKernel.shape[0],j:j+flpKernel.shape[1]];
            # if i<3 and j<3:
            #     print(i,j,tmp.shape,tmp,tmp*flpKernel,np.sum(np.sum(tmp*flpKernel)));
            output[i,j] = np.sum(np.sum(tmp*flpKernel));
            #print(tmp,tmp*kernel[:tmp.shape[0],:tmp.shape[1]],output[i,j]);
    #tmp = scipy.ndimage.convolve(image,kernel,mode='constant');
    #print(np.fabs(tmp-output)<1e-9);
    return output


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([[1,0,-1]])  # 1 x 3
    ky = np.array([[1],[0],[-1]])  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2+Iy**2);

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]);
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]);

    Gx = convolve(image,kx);
    Gy = convolve(image,ky);

    grad_magnitude = np.sqrt(Gx**2+Gy**2);

    return Gx, Gy, grad_magnitude

def gaussian_func(sigma,x,y):
    return np.exp(-(x**2+y**2)/(2*(sigma**2)))/(2*np.pi*(sigma**2));

def bilateral_filter(image, window_size, sigma_d, sigma_r):
    """
    Return filtered image using a bilateral filter

    Input-  image: H x W
            window_size: (h, w)
            sigma_d: sigma for the spatial kernel
            sigma_r: sigma for the range kernel
    Output- output: filtered image
    """
    # TODO: complete the bilateral filtering, assuming spatial and range kernels are gaussian

    output    = np.zeros(image.shape);
    h_2       = math.floor(window_size[0]/2.0);
    w_2       = math.floor(window_size[1]/2.0);
    kernel_d  = np.resize(np.array([(i**2+j**2)/(2*(sigma_d**2)) for i in range(-h_2,h_2+1,1) for j in range(-w_2,w_2+1,1)]),window_size);
    padImage  = np.zeros((image.shape[0]+2*h_2,image.shape[1]+2*w_2));
    padImage[h_2:h_2+image.shape[0],w_2:w_2+image.shape[1]] = image;
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            tmp = padImage[i:i+window_size[0],j:j+window_size[1]];
            tmpKernel = np.exp(-kernel_d-np.fabs(tmp-image[i,j])**2/(2*(sigma_r**2)));
            output[i,j] = np.sum(np.sum(tmpKernel*tmp))/np.sum(np.sum(tmpKernel));

    return output


def main():
    # The main function
    img = read_img('./SEM.png')
    """ Image Patches """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")
    '''
    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.hstack((patches[0],patches[1],patches[2]));
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code
    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")
    
    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    # kernel_gaussian = np.array([[0.0228906061, 0.1055219196, 0.0228906061],\
    #                     [0.1055219196, 0.4864386495, 0.1055219196],\
    #                     [0.0228906061, 0.1055219196, 0.0228906061]]);
    kernel_gaussian = np.resize(np.array([gaussian_func(0.572,i,j) for i in range(-1,2,1) for j in range(-1,2,1)]),(3,3));
    # print(np.fabs(kernel_gaussian-kernel_gaussian1)<1e-6);
    # print(kernel_gaussian1);

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    kernel_gaussian = np.resize(np.array([gaussian_func(3,i,j) for i in range(-2,3,1) for j in range(-2,3,1)]),(5,5));
    # print(np.fabs(kernel_gaussian-kernel_gaussian1)<1e-6);
    # print(kernel_gaussian1);

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/taks2bil_gaussian.png")
    # blurred_img = scipy.ndimage.gaussian_filter(img,sigma=0.527,radius=1,mode='constant',cval=0.0);
    # print(np.fabs(fliterImage-blurred_img)<1e-1);

    # (d), (e): No code

    # (f): Complete edge_detection()
    '''
    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    edge_detect[edge_detect<0.5] = 0;
    save_img(edge_detect, "./gaussian_filter/edge.png")
    # _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    # save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    '''
    # (h) complete biliateral_filter()
    if not os.path.exists("./bilateral"):
        os.makedirs("./bilateral")

    image_bilataral_filtered = bilateral_filter(img, (5, 5), 3, 75)
    save_img(image_bilataral_filtered, "./bilateral/bilateral_output.png")
#newimg = cv2.imread("./grace_hopper.png",0);
#tmpImg = np.array(cv2.bilateralFilter(newimg,d=2,sigmaColor=75,sigmaSpace=3))
#save_img(tmpImg,"./bilateral/ans.png");
#print(image_bilataral_filtered);
#print(tmpImg);
#print(np.fabs(image_bilataral_filtered-tmpImg)<1e-3);

    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img,kernel_LoG1)
    filtered_LoG2 = convolve(img,kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")
    
    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    print("LoG Filter is done. ")
    data = np.load('log1d.npz')
    plt.plot(data['gauss53']-data['gauss50']);
    plt.show()
    # save_img(data['gauss50'],"./log_filter/gauss50.png")
'''

if __name__ == "__main__":
    main()
