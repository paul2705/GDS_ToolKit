"""
Homography fitting functions
You should write these
"""
import numpy as np
from common import homography_transform

def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    n,m = XY.shape;
    x = np.zeros((2*n,9));
    y = np.zeros((2*n));
    for i in range(n):
        x[2*i,:2] = -XY[i,:2]; x[2*i,2] = -1; 
        x[2*i,6] = XY[i,0]*XY[i,2]; x[2*i,7] = XY[i,1]*XY[i,2]; x[2*i,8] = XY[i,2];
        x[2*i+1,3:5] = -XY[i,:2]; x[2*i+1,5] = -1;
        x[2*i+1,6] = XY[i,0]*XY[i,3]; x[2*i+1,7] = XY[i,1]*XY[i,3]; x[2*i+1,8] = XY[i,3];
#        y[2*i] = XY[i,2];
#        y[2*i+1] = XY[i,3];
    U,S,V = np.linalg.svd(x);
    H = V[8];
    return H/H[8];


def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    for iter in range(nIters):
        subset = np.random.choice(XY.shape[0],4,replace=False);
        H = fit_homography(XY[subset]);
        H = np.reshape(H/H[8],(3,3));
        new_point = homography_transform(XY[:,:2],H);
        del_x = (XY[:,2:]-new_point)[:,0];
        del_y = (XY[:,2:]-new_point)[:,1];
        distance = np.sqrt(del_x**2 + del_y**2);
        inlier = distance<eps;
        if np.sum(np.sum(inlier))>bestCount:
            bestH, bestCount, bestInliers = H, np.sum(np.sum(inlier)), inlier;
    bestRefit = fit_homography(XY[bestInliers]);
    return bestRefit

if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    pass
