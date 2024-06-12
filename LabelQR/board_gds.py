import nazca as nd
import nazca.geometries as geom
from QRCode import encoderStandard, decoderStandard
from QRCode import encoderMicro, decoderMicro

'''
GDS
Alignment: 0
Si grating EBL: 1
Si waveguide: 2
SiN waveguide: 3
SiN EBL: 4
'''
# ====================================Encoder Exmaple================================
'''
You can use encoder.makeLabelWithQRCode() to create label text with its corresponding Standard QRCode or Micro QRCode
The usage of encoder.makeLabelWithQRCode() is similar to nd.text()
For details of parameters meanings please refer to ./QRCode/encoderStandard.py or ./QRCode/encoderMicro.py
'''
# encoderStandard.makeQRCode(text='D50S150WG6',height=0.2,align='cc',QRSize=0.8,layer=2).put(600)
# for i in range(-7000, 7000, 1000):
#     for j in range(-5000, 0000, 1000):
#         encoderStandard.makeQRCode(text='@({:d}, {:d})'.format(i,j),height=10*(((i+7000)/1000)%4+1),align='cc',layer=0).put(i, j)
# 500 nm resolution
# vary size
encoderStandard.makeQRCode(text='@({:d}, {:d})'.format(0,0),height=10*(((0+7000)/1000)%4+1),align='cc',layer=0).put(0,0)

# Bar Code

# View Size: [1cm*1cm] -> 1mm*1mm
nd.export_gds()
'''
Optimal Fibrer target point 
photo detector 
moving fiber -> target point Gaussian 
Rotation fiber
'''
# ====================================Decoder Exmaple================================
'''
You can use decoder.readQRCode() to detect text (label) from Standard QRCode or Micro QRCode in the specified image
Use several different method to detect texts from QRCodes
For details of parameters meanings please refer to ./QRCode/decoderStandard.py or ./QRCode/decoderMicro.py
'''

# print('==============================exmaple 1==============================')
# print(decoderStandard.readQRCode("./QRCode/example_pos_1.png"))
# print('==============================exmaple 2==============================')
# print(decoderStandard.readQRCode("./QRCode/view_example.png"))