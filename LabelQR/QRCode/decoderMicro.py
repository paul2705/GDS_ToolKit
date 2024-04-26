import numpy as np
import pyboof as pb

def readQRCode(ImgPath, KlayoutDecode = 1):
    """Read QRCode included in the screenshot of Klayout / Other view (not yet implement) 
    Use several different methods to detect QRCode to increase the hit rate.
   
    Args:
        ImgPath (String): the file path of the Image that needs to read QRCode
        
    Returns:
        detected_text (List): a list of detected text from QRCodes. If no valid QRCode detected, return None
    """
    # Create a QReader instance
    detector = pb.FactoryFiducial(np.uint16).microqr()

    # Get the image that contains the QR code
    image = pb.load_single_band(ImgPath, np.uint16)

    # Use the detect_and_decode function to get the decoded QR data
    detector.detect(image)
    ret = [qr.message for qr in detector.detections]
    if len(ret) >= 1:
        print("Success!")
        return ret
    else:
        print("ERROR: Decode QR Code Failed!")

    return None
