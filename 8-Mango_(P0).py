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
for i in range(170):
    image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Mango_(P0)\\image_ ('+str(i+1)+').jpg')
    median = cv2.medianBlur(image, 3)
    #================================================================ HSV color thresholding
    result = median.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for green color
    lower1 = np.array([36, 0, 0])
    upper1 = np.array([70, 255, 255])
    # upper boundary for yellow color
    lower2 = np.array([15, 0, 0])
    upper2 = np.array([36, 255, 255])
    lower_mask = cv2.inRange(image1, lower2, upper2)
    upper_mask = cv2.inRange(image1, lower1, upper1)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(result, result, mask=full_mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #===============================================================
    eroded = cv2.erode(gray, (5, 5), iterations=1)
    (T, threshInv) = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opening)
    median2 = cv2.medianBlur(opening, 11)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\8-Mango_(P0)\\'+str(i+1)+'.jpg', median2)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\8-Mango_(P0)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Mango_(P0)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)
    simxy = jaccard_binary(img_np, img2_np)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/170)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()