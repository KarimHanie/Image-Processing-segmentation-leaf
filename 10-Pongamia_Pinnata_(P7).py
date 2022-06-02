
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
for i in range(319):
    image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Pongamia_Pinnata_(P7)\\image_ ('+str(i+1)+').jpg')
    median = cv2.medianBlur(image, 15)
    out = cv2.addWeighted(median, 3, median, 0, 10)
    #================================================================ HSV color thresholding
    result = median.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Lower boundary for yellow color
    lower1 = np.array([15, 0, 0])
    upper1 = np.array([36, 255, 255])
    # Upper boundary for green color
    lower2 = np.array([36, 0, 0])
    upper2 = np.array([70, 255, 255])
    lower_mask = cv2.inRange(image1, lower1, upper1)
    upper_mask = cv2.inRange(image1, lower2, upper2)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(result, result, mask=full_mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #===============================================================
    eroded = cv2.erode(gray, (5, 5), iterations=1)
    # ========================================= closing ===============================
    (T, threshInv2) = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\10-Pongamia_Pinnata_(P7)\\'+str(i+1)+'.jpg', closing)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\10-Pongamia_Pinnata_(P7)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Pongamia_Pinnata_(P7)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)
    simxy = jaccard_binary(img_np, img2_np)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/319)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()