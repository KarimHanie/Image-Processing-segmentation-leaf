import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
def jaccard_binary(x,y):
    interSec=np.logical_and(x,y)
    union =np.logical_or(x,y)
    similarity =interSec.sum() /float(union.sum())
    return similarity
sum_per=0.0
for i in range(174):
    image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Alstonia_Scholaris_(P2)\\image_ ('+str(i+1)+').jpg')
    median = cv2.medianBlur(image, 3)
    #================================================================ HSV color thresholding
    result = median.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for yellow color
    lower2 = np.array([30, 20, 0])
    upper2 = np.array([40, 255, 255])
    # upper boundary for green color
    lower1 = np.array([41, 0, 0])
    upper1 = np.array([70, 255, 255])
    lower_mask = cv2.inRange(image1, lower2, upper2)
    upper_mask = cv2.inRange(image1, lower1, upper1)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(result, result, mask=full_mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #===============================================================
    eroded = cv2.erode(gray, (5, 5), iterations=1)
    # cv2.imshow('eroded', eroded)
    (T, threshInv) = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opening)
    (T2, threshInv2) = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    median2 = cv2.medianBlur(closing, 15)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\1-Alstonia_Scholaris_(P2)\\'+str(i+1)+'.jpg', median2)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\1-Alstonia_Scholaris_(P2)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Alstonia_Scholaris_(P2)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)
    simxy = jaccard_binary(img_np, img2_np)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/174)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()