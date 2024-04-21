"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img, get_AKAZE
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2
from scipy import ndimage

def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    dist = np.sum(desc1**2,axis=1,keepdims=True) + np.sum(desc2**2,axis=1) - 2*np.dot(desc1,desc2.T);
    return dist

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches 
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    dist = compute_distance(desc1,desc2);
    N,M = dist.shape;
    matches = [];
    idx = np.argsort(dist,axis=1);
    dist = np.take_along_axis(dist,idx,axis=1);
    print(dist);
    for i in range(N):
        if dist[i,0]/dist[i,1]<ratioThreshold:
            matches.append([i,idx[i,0]]);
    print(matches);
    return np.array(matches)

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints
    Hint: see cv2.line
    '''
    #Hint:
    #Use common.get_match_points() to extract keypoint locations
    XY = common.get_match_points(kp1,kp2,matches);
    print(img2.shape);
    N1, M1, _ = img1.shape;
    N2, M2, _ = img2.shape;
    # N1, M1 = img1.shape;
    # N2, M2 = img2.shape;
    newImg = np.zeros((N1+N2,max(M1,M2),3));
    newImg[:N1,:M1,:] = img1;
    newImg[N1:N1+N2,:M2,:] = img2;
    for i in range(len(XY)):
        cv2.line(newImg, (int(XY[i,0]),int(XY[i,1])), (int(XY[i,2]),int(XY[i,3]+N1)), (0,255,0), 3);
    return newImg


def warp_and_combine(img1, img2, H):
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
    '''
    N1, M1, _ = img1.shape;
    N2, M2, _ = img2.shape;
    cornor1 = np.array([[0,0],[M1,0],[0,N1],[M1,N1]]);
    cornor2 = np.array([[0,0],[M2,0],[0,N2],[M2,N2]]);
    cornor1T = common.homography_transform(cornor1,H);
    cornor = np.vstack((cornor2,cornor1T,cornor1));
    x_min = int(np.floor(min(cornor[:,0])));
    x_max = int(np.ceil(max(cornor[:,0])));
    y_min = int(np.floor(min(cornor[:,1])));
    y_max = int(np.ceil(max(cornor[:,1])));
    delta = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]]).astype(np.float32);
    newSize = ((x_max-x_min),(y_max-y_min));
    img1T = cv2.warpPerspective(img1,np.dot(delta,H),newSize,flags=cv2.INTER_LINEAR)
    img2T = cv2.warpPerspective(img2,delta,newSize,flags=cv2.INTER_LINEAR)
    newImg = np.zeros((newSize[1],newSize[0],3));
    IsColor = 1;

    if IsColor:
        Alpha=((img1T[:,:,0]+img1T[:,:,1]+img1T[:,:,2])>0);
        Beta =((img2T[:,:,0]+img2T[:,:,1]+img2T[:,:,2])>0);
        img1T = np.array(img1T).astype(np.float64);
        img2T = np.array(img2T).astype(np.float64);
        for col in range(3):
            newImg[:,:,col]=img1T[:,:,col]*Alpha*(1-Alpha*Beta)+img2T[:,:,col]*Beta*(1-Alpha*Beta)+(img1T[:,:,col]+img2T[:,:,col])/2.0*Alpha*Beta;
            # newImg[:,:,col]=img1T[:,:,col]*Alpha + img2T[:,:,col]*(1-Alpha);
    else:
        Alpha=(img1T>0);
        newImg=img1T*Alpha+img2T*(1-Alpha);
        
    return newImg;


def make_warped(img1, img2, Padding=1000, Delta=200):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
    
    Output - Final stitched image
    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''
    FromImg = img1; ToImg = img2;
    kp1, desc1 = get_AKAZE(img1);
    kp2, desc2 = get_AKAZE(img2);
    XY = common.get_match_points(kp1,kp2,find_matches(desc1,desc2,0.8));
    H = RANSAC_fit_homography(XY);
    H = np.reshape(H/H[8],(3,3));
    
    return warp_and_combine(img1,img2,H);


if __name__ == "__main__":

    #Possible starter code; you might want to loop over the task 6 images
    I1 = read_img(os.path.join('p2.jpg'))
    I2 = read_img(os.path.join('p1.jpg'))
    kp1, desc1 = get_AKAZE(I1);
    kp2, desc2 = get_AKAZE(I2);
#res = draw_matches(I1,I2,kp1,kp2,find_matches(desc1,desc2,0.8));
    res = make_warped(I1,I2);
    save_img(res,"result_"+to_stitch+".jpg")
