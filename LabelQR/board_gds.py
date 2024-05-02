import nazca as nd
import nazca.geometries as geom
from QRCode import encoderStandard, decoderStandard
from QRCode import encoderMicro, decoderMicro

# ====================================Encoder Exmaple================================
'''
You can use encoder.makeLabelWithQRCode() to create label text with its corresponding Standard QRCode or Micro QRCode
The usage of encoder.makeLabelWithQRCode() is similar to nd.text()
For details of parameters meanings please refer to ./QRCode/encoderStandard.py or ./QRCode/encoderMicro.py
'''
# encoderStandard.makeQRCode(text='D50S150WG6',height=0.2,align='cc',QRSize=0.8,layer=2).put(600)
for i in range(-5000, 5000, 1000):
    for j in range(-5000, 5000, 1000):
        encoderStandard.makeQRCode(text='@({:d}, {:d})'.format(i,j),height=40,align='cc',layer=2).put(i, j)

# Bar Code

# View Size: [1cm*1cm] -> 1mm*1mm
nd.export_gds()


# ====================================Decoder Exmaple================================
'''
You can use decoder.readQRCode() to detect text (label) from Standard QRCode or Micro QRCode in the specified image
Use several different method to detect texts from QRCodes
For details of parameters meanings please refer to ./QRCode/decoderStandard.py or ./QRCode/decoderMicro.py
'''

print('==============================exmaple 1==============================')
print(decoderStandard.readQRCode("./QRCode/example_pos_1.png"))
print('==============================exmaple 2==============================')
print(decoderStandard.readQRCode("./QRCode/view_example.png"))