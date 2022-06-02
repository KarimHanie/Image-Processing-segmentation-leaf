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
for i in range(214):
    image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Arjun_(P1)\\image_ ('+str(i+1)+').jpg', 0)
    median = cv2.medianBlur(image,15)
    out = cv2.addWeighted(median, 3, median, 0, -50)

    (T, threshInv) = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\2-Arjun_(P1)\\'+str(i+1)+'.jpg', opening)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\2-Arjun_(P1)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Arjun_(P1)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)

    simxy = jaccard_binary(img_np, img2_np)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/214)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()