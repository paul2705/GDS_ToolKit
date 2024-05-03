from phidl import Device
import nazca as nd

def convertPhidlToNAZCA(phidlDevice,layer=20):
    """An Interface converting phidl to nazca
   
    Args:
        phidlDevice (phidl.Device): the phidl Device that contains all the polygons needed to convert to NAZCA Cell
        (Optional) layer (Int): layer reference to generate label and QRCode, default will be 20

    Returns:
        Cell: flattened NAZCA cell with converted phidl polygons.
    """
    polygons = phidlDevice.get_polygons(by_spec=False)
    with nd.Cell(f'label_phidl') as convertedCell:
        startPoint = (0, 0)
        for polygon in polygons:
            nd.Polygon(layer=layer, points=polygon).put(startPoint)

    return convertedCell
