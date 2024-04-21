import sift
from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters

ImgName='../../Image/test01.png';
Img1=array(Image.open(ImgName).convert('L'));
# NewImg1=filters.gaussian_filter(Img1[:,:],2);
# imsave("../../Image/test01_gauss.png",NewImg1);
# ImgName='../../Image/test01_gauss.png';
sift.ProcessImage(ImgName,'../../Image/large.sift',"--edge-thresh 10 --peak-thresh 5 -O 2");
l1,d1=sift.ReadFeaturesFromFile('../../Image/large.sift');

ImgName='../../Image/template0.png';
Img2=array(Image.open(ImgName).convert('L'));
# NewImg2=filters.gaussian_filter(Img2[:,:],8);
# imsave("../../Image/template0_gauss.png",NewImg2);
# ImgName='../../Image/template0_gauss.png';
sift.ProcessImage(ImgName,'../../Image/small.sift',"--edge-thresh 10 --peak-thresh 5 -O 5");
l2,d2=sift.ReadFeaturesFromFile('../../Image/small.sift');

print('Starting Matching...');
Matches=sift.MatchTwoSided(d1,d2);

figure(); gray();
sift.PlotMatches(Img1,Img2,l1,l2,Matches,False);
savefig('../../Image/match.jpg')
show();
