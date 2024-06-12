# from QRCode import encoderStandard, decoderStandard
# from QRCode import encoderMicro, decoderMicro

from qreader import QReader
from PIL import Image
import numpy as np

def readQRCode(ImgPath, KlayoutDecode = 1):
    """Read QRCode included in the screenshot of Klayout / Other view (not yet implement) 
    Use several different methods to detect QRCode to increase the hit rate.
   
    Args:
        ImgPath (String): the file path of the Image that needs to read QRCode
        (Optional) KlayoutDecode: If it is Klayout View (KlayoutDecode=1), we implement some method to detect QRCode. Default: 1

    Returns:
        detected_text (tuple): a tuple of detected text from QRCodes. If no valid QRCode detected, return None
    """
    # Create a QReader instance
    qreader = QReader(model_size='s')

    # Get the image that contains the QR code
    # image = cv2.cvtColor(cv2.imread(ImgPath), cv2.COLOR_BGR2RGB)
    image = np.array(Image.open(ImgPath).convert('RGB'))
    
    return qreader.detect_and_decode(image=image, return_detections = True)

def calculateRotation(quadPoints):
    # Assuming quad_points is a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # Calculate the angle of rotation using the first two points
    p1, p2 = quadPoints[0,:], quadPoints[1,:]
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    angle1 = np.arctan2(deltaY, deltaX) * (180.0 / np.pi)
    angle1 = 180+angle1 if angle1<0 else angle1

    p1, p2 = quadPoints[2,:], quadPoints[3,:]
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    angle2 = np.arctan2(deltaY, deltaX) * (180.0 / np.pi)
    angle2 = 180+angle2 if angle2<0 else angle2
    print(angle1, angle2)
    return (angle1+angle2)/2.0


if __name__=='__main__':
    ret = readQRCode('../QRCode/example9_29_2.png')
    print(len(ret[1]),ret[0])
    rotationAngle = 0
    cnt = 0
    for i in range(len(ret[1])):
        if (ret[0][i]==None):
            continue
        cnt += 1
        print(ret[0][i],ret[1][i]['confidence'])
        rotationAngle += calculateRotation(ret[1][i]['quad_xy'])
    print(rotationAngle/cnt)