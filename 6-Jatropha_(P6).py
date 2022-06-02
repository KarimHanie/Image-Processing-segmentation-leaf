import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img


def jaccard_binary(x,y):
    interSec=np.logical_and(x,y)
    union =np.logical_or(x,y)
    similarity =interSec.sum() /float(union.sum())
    return similarity
sum_per=0.0
for i in range(131):
    image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Jatropha_(P6)\\image_ ('+str(i+1)+').jpg')
    img2 = hisEqulColor(image)
    dst = cv2.equalizeHist(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    col=cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved_gray_histogram\\'+str(i+1)+'.jpg',img2)

    out = cv2.addWeighted(img2, 1, img2, 0, -55)
    gaussian_blur = cv2.GaussianBlur(src=out, ksize=(3, 3), sigmaX=5, sigmaY=5)

    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharped_img = cv2.filter2D(gaussian_blur, -1, sharpen_filter)
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved\\'+str(i+1)+'.jpg', sharped_img)
    # ============================================== color mask HSV
    result = sharped_img.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for green color
    lower1 = np.array([40, 0, 0])
    upper1 = np.array([65, 255, 255])
    # upper boundary for yellow color
    lower2 = np.array([66, 0, 0])
    upper2 = np.array([80, 255, 255])
    lower_mask = cv2.inRange(image1, lower1, upper1)
    upper_mask = cv2.inRange(image1, lower2, upper2)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(result, result, mask=full_mask)
    # =================================================================
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # ========================================================== opening and closing
    (T2, threshInv2) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
# ============================================================ to remove the rest of the noise
    median2 = cv2.medianBlur(opening2, 11)
    ero2 = cv2.erode(median2, (10, 10), iterations=3)
    dilated = cv2.dilate(ero2.copy(), None, iterations=3)
    #============================================================== Closing
    (T3, threshInv3) = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closing3 = cv2.morphologyEx(threshInv3, cv2.MORPH_CLOSE, kernel)


    # cv2.imshow('opening and closing_median  ', median)

    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\6-Jatropha_(P6)\\'+str(i+1)+'.jpg', closing3)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\6-Jatropha_(P6)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Jatropha_(P6)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)

    # print(img_np.shape)
    # print(img2_np.shape)

    simxy = jaccard_binary(img_np, img2_np)
    # print(simxy)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/131)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()