"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img, get_AKAZE
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2
from scipy import ndimage
from task6 import *

def task7_warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.
    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them
    Output - V: stitched image of size (?,?,3); unknown since it depends on H
                but make sure in V, for pixels covered by both img1 and warped img2,
                you see only img2
    '''
    img1T = cv2.warpPerspective(img1,H,(img2.shape[1],img2.shape[0]),flags=cv2.INTER_LINEAR);
    img2T = np.zeros(img1T.shape);
    img2T = img2;
    newImg = np.zeros(img1T.shape);
    IsColor = 1;

    if IsColor:
        Alpha=((img1T[:,:,0]+img1T[:,:,1]+img1T[:,:,2])>0);
        Beta =((img2T[:,:,0]+img2T[:,:,1]+img2T[:,:,2])>0);
        for col in range(3):
            newImg[:,:,col]=img1T[:,:,col]*Alpha*(1-Alpha*Beta)+img2T[:,:,col]*Beta*(1-Alpha*Beta)+(img1T[:,:,col]+img2T[:,:,col])/2.0*Alpha*Beta;
            # newImg[:,:,col]=img1T[:,:,col]*Alpha + img2T[:,:,col]*(1-Alpha);
    else:
        Alpha=(img1T>0);
        newImg=img1T*Alpha+img2T*(1-Alpha);
        
    return newImg;

def improve_image(scene, template, transfer):
    '''
    Detect template image in the scene image and replace it with transfer image.

    Input - scene: image (H,W,3)
            template: image (K,K,3)
            transfer: image (L,L,3)
    Output - augment: the image with 
    
    Hints:
    a) You may assume that the template and transfer are both squares.
    b) This will work better if you find a nearest neighbor for every template
       keypoint as opposed to the opposite, but be careful about directions of the
       estimated homography and warping!
    '''
    kp1, desc1 = get_AKAZE(template);
    kp2, desc2 = get_AKAZE(scene);
    XY = common.get_match_points(kp1,kp2,find_matches(desc1,desc2,0.8));
    H = RANSAC_fit_homography(XY);
    H = np.reshape(H/H[8],(3,3));
    newTrans = np.zeros(template.shape);
    for i in range(3):
        newTrans[:,:,i] = cv2.resize(transfer[:,:,i],(template.shape[1],template.shape[0]),interpolation = cv2.INTER_LINEAR);
    # return draw_matches(template,scene,kp1,kp2,find_matches(desc1,desc2,0.8));
    return task7_warp_and_combine(newTrans,scene,H);

if __name__ == "__main__":
    # Task 7
    to_stitch = 'bbb'
    I1 = read_img(os.path.join('./gaussian_filter/edge01.png'))
    I2 = read_img(os.path.join('./gaussian_filter/slice02.png'))
    print(I2.shape);
    # I3 = read_img(os.path.join('task7/seals/','um.png'))
    # print(I1.shape,I2.shape,I3.shape);
    
    kp1, desc1 = get_AKAZE(I2);
    kp2, desc2 = get_AKAZE(I1);
    print(kp1);
    res = draw_matches(I2,I1,kp1,kp2,find_matches(desc1,desc2,50));
    # res = improve_image(I1,I2,I3);
    # save_img(res,"result_"+to_stitch+".jpg")
    save_img(res,"./gaussian_filter/match02.jpg")
    pass
