from PIL import Image
from numpy import *
from pylab import *

ImgName='../../Image/test01.png';
Img1=array(Image.open(ImgName).convert('L'));

