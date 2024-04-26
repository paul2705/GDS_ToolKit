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
    
    if KlayoutDecode == 0:
        # Use the detect_and_decode function to get the decoded QR data
        return qreader.detect_and_decode(image=image)

    if KlayoutDecode == 1:
        N, M, C = image.shape
        # newImg = image[::]
        print("Origin Image Attempt: 0")
        decoded_text = qreader.detect_and_decode(image=image)
        if (len(decoded_text)>0 and decoded_text[0]!=None):
            print("Success!")
            return decoded_text

        highColor = (np.unique(image[:,:,0])[0],np.unique(image[:,:,1])[0],np.unique(image[:,:,2])[0])
        highImage = image.copy()
        for i in range(N):
            for j in range(M):
                if (highImage[i,j,0]==128 and highImage[i,j,1]==128 and highImage[i,j,2]==128):
                    highImage[i,j,:] = 255
                if np.sum(highImage[i,j])<255*3:
                    highImage[i,j,:] = highColor[:]
        
        newImg = highImage.copy()
        for _ in range(180,40,-20):
            print(f'Highly Hit Attempt: {((180-_)//20)+1}')
            tmpImg = newImg[::]
            tmpImgRet = __fillHoles(tmpImg,highColor,_)
            # saveImage = Image.fromarray(tmpImgRet)
            # saveImage.save(f'testh_{_}.png')
            # cv2.imwrite(f'testh_{_}.png',tmpImgRet)
            decoded_text = qreader.detect_and_decode(image=tmpImgRet)
            if (len(decoded_text)>0 and decoded_text[0]!=None):
                print("Success!")
                return decoded_text

        newImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for _ in range(10):
            print(f"Resize Image Attempt: {_+1}")
            try:
                newImg = cv2.resize(newImg, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
                # saveImage = Image.fromarray(tmpImgRet)
                # saveImage.save(f'testh_{_}.png')
                # cv2.imwrite(f"test_{_+1}.png",newImg)
                decoded_text = qreader.detect_and_decode(image=newImg)
                if (len(decoded_text)>0 and decoded_text[0]!=None):
                    print("Success!")
                    return decoded_text
            except:
                break
        color = (np.unique(image[:,:,0])[0],np.unique(image[:,:,1])[0],np.unique(image[:,:,2])[0])
        for i in range(N):
            for j in range(M):
                if (image[i,j,0]==128 and image[i,j,1]==128 and image[i,j,2]==128):
                    image[i,j,:] = 255
                if np.sum(image[i,j])<255*3:
                    image[i,j,:] = color[:]
        
        newImg = image[::]
        for _ in range(0,20,5):
            print(f'Fill Holes Image Attempt: {_//5+1}')
            tmpImg = newImg[::]
            tmpImgRet = __fillHoles(tmpImg,color,_)
            # saveImage = Image.fromarray(tmpImgRet)
            # saveImage.save(f'testh_{_}.png')
            # cv2.imwrite(f'testh_{_}.png',tmpImgRet)
            decoded_text = qreader.detect_and_decode(image=tmpImgRet)
            if (len(decoded_text)>0 and decoded_text[0]!=None):
                print("Success!")
                return decoded_text
        for _ in range(20,200,20):
            print(f'Fill Holes Image Attempt: {(_//20)+4}')
            tmpImg = newImg[::]
            tmpImgRet = __fillHoles(tmpImg,color,_)
            # saveImage = Image.fromarray(tmpImgRet)
            # saveImage.save(f'testh_{_}.png')
            # cv2.imwrite(f'testh_{_}.png',tmpImgRet)
            decoded_text = qreader.detect_and_decode(image=tmpImgRet)
            if (len(decoded_text)>0 and decoded_text[0]!=None):
                print("Success!")
                return decoded_text

        newImg = image[::]
        for _ in range(10):
            print(f'Flood Fill Image Attempt: {_+1}')
            image[::] = newImg[::]
            for i in range(N):
                for j in range(M):
                    if np.sum(image[i,j])<255*3:
                        for l in range(np.max([0,i-1]),np.min([N,i+1]),1):
                            for r in range(np.max([0,j-1]),np.min([M,j+1]),1):
                                newImg[l,r,:] = image[i,j,:]
            
            decoded_text = qreader.detect_and_decode(image=newImg)
            if (len(decoded_text)>0 and decoded_text[0]!=None):
                print("Success!")
                return decoded_text
            # tmpImg = cv2.GaussianBlur(newImg, (2*int(12/(_+2))+1, 2*int(12/(_+2))+1), _)
            tmpImg = newImg[::]
            tmpImgRet = __fillHoles(tmpImg,color,80)
            # saveImage = Image.fromarray(tmpImgRet)
            # saveImage.save(f'testh_{_}.png')
            # cv2.imwrite(f'testh_{_}.png',tmpImgRet)
            decoded_text = qreader.detect_and_decode(image=tmpImgRet)
            if (len(decoded_text)>0 and decoded_text[0]!=None):
                print("Success!")
                return decoded_text

    print("ERROR: Decode QR Code Failed!")
    return None

def __fillHoles(image,color,threshold):
    N, M, C = image.shape
    # newImage = image.copy()
    # newImage[image<255] = 255
    visited = np.zeros((N,M))
    tmpVis = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if (visited[i,j]==1):
                continue
            tmpVis[::] = visited[::]
            cnt = __bfs(i,j,image,visited,fill=0)
            if (cnt<threshold and cnt>0):
                visited[::] = tmpVis[::]
                __bfs(i,j,image,visited,fill=1,color=color,target=image)
    
    return image
    

def __bfs(i,j,image,visited,fill=0,color=(255,255,255),target=None):
    if np.sum(image[i,j])!=255*3:
        return 0
        
    N, M, _ = image.shape
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    queue = [] 
    cnt = 1
    queue.append((i,j))
    visited[i,j] = True
 
    while queue:
        x,y = queue.pop(0)
        if (fill==1 and np.sum(image[x,y])==255*3):
            target[x,y,:] = color
        for _ in range(4):
            nx = x + dx[_]
            ny = y + dy[_]
            if (nx>=0 and ny>=0 and nx<N and ny<M):
                if (visited[nx,ny]==1):
                    continue
                if (np.sum(image[nx,ny])==255*3):
                    cnt = cnt + 1
                    visited[nx,ny] = True
                    queue.append((nx,ny))
    
    return cnt