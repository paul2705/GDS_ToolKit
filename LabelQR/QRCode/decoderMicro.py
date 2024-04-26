import numpy as np
import pyboof as pb

def readQRCode(ImgPath, KlayoutDecode = 1):
    """Read QRCode included in the screenshot of Klayout / Other view (not yet implement) 
    Use several different methods to detect QRCode to increase the hit rate.
   
    Args:
        ImgPath (String): the file path of the Image that needs to read QRCode
        
    Returns:
        detected_text (list): a list of detected text from QRCodes. If no valid QRCode detected, return Empty List
    """
    # Create a QReader instance
    detector = pb.FactoryFiducial(np.uint16).microqr()

    # Get the image that contains the QR code
    image = pb.load_single_band(ImgPath, np.uint16)
    print(image)

    # Use the detect_and_decode function to get the decoded QR data
    detector.detect(image)
    print(len(detector.detections))
    ret = [qr.message for qr in detector.detections]
    return ret
