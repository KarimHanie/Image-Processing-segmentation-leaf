# Image-Processing-segmentation-leaf

## 1-`Alstonia_Scholaris_(P2)`

![1.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb42fc3f-f032-46b1-9357-2bb4f8be715b/1.jpg)

## part 1:

```python
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
```

 

> (1)
I will read file form the saved Location using `cv2.imread` , then I used Median Blur to reduce Noise on image 
using `medianBlur` method that takes Src image and the value of matrix i want to apply on it in our case its a 3x3 matrix 
(2)
after we remove the noise from the image we want to target our leaf so i used **full mask threshold** to target green color 
First,  we detect the lower and upper boundary of space colors we want to limit our search on them  using  guiding image .
(3) 
after we combined the lower and upper boundary we generate new image with the only target Colors values
> 

  

# Guiding Image to get the values of H,S,V
![unknown](https://user-images.githubusercontent.com/85907989/171634212-32cfa9bb-4a87-4bc7-86e6-5f5cf3449223.png)



# part 2:

```python
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #===============================================================
    eroded = cv2.erode(gray, (5, 5), iterations=1)
    # cv2.imshow('eroded', eroded)
   # Opening 
    (T, threshInv) = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opening)
    (T2, threshInv2) = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    median2 = cv2.medianBlur(closing, 15)
```

> (1)
then I changed the RGB color of image into gray we can apply it on `Opening and Closing  OTSU threshold Method` 
after the opening and closing method some images get holes in side it  like image below
> 

![12.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a3a3d29-50d2-4fce-b0cc-38ef206664c6/12.jpg)

> (2) 
that’s why I used the `closing method` after that so i can close these holes 
(3)
also as shown above there still exists some noise in the image that’s why I used `MedianBlur` again before passing  the image to `jaccard` function to compare it with the source image
> 

# part 3

```python
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
```

> (1)
 take median image in the code above ‘ line above ’ and save it into folder to see the how my result look like and what i need to enhance in the next run 
 and take the image from the file and the original image Groundtruth save both in two variables and pass them to `np.array` 
to convert image into Numpy array
(2)
after that i send the value of 2 arrays to jaccard method that flatten these array and compare each pixel  to another and calculate the similarity 
(3)
sum all these values and Calculate the total percentage of folder
> 

---

# 2- folders : `Arjun_(P1)` ,`Basil_(P8)`, `Chinar_(P11)`

> these folders are similar in the approach taken to enhance images
> 

```python
image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Arjun_(P1)\\image_ ('+str(i+1)+').jpg', 0)
    median = cv2.medianBlur(image,15)
    out = cv2.addWeighted(median, 3, median, 0, -50)
    dilated = cv2.dilate(out.copy(), None, iterations=3)

    eroded = cv2.erode(dilated, (5, 5), iterations=2)
```

# `Arjun_(P1)`

![2.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3c22ec88-c61c-47cb-b114-927132595cd6/2.jpg)

```python
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
```

> after i read the file ⇒ apply median method on the image to remove the noise from it then 
i used `addWighted method =>` used 4 inputs `src1=median, alpha=3,src2=median,beta=0 ,gamma=-130`
this function work ⇒ choose 2 images and apply them together to get new enhanced one and the gamma to change the brightness of the image if the value was <0 then it will change to a darker scale if it was > 0 then it will change to a brighter image ….
(2) 
then i apply the OTSU threshold to open and close the image 
(3) send the new generated image to `jaccard` method to do the comparison as we explain above
> 

# `Basil_(P8)`

![3.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/011c81b5-80aa-4c16-b1ac-8616445f184e/3.jpg)

```python
image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Basil_(P8)\\image_ ('+str(i+1)+').jpg', 0)
    median = cv2.medianBlur(image, 3)
    out = cv2.addWeighted(median, 4, median, 0, -30)
    (T, threshInv) = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\3-Basil_(P8)\\'+str(i+1)+'.jpg', opening)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\3-Basil_(P8)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Basil_(P8)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)
    simxy = jaccard_binary(img_np, img2_np)
    # print(simxy)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/148)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> same as above except here image was so bright so we change the gamma to darker scale that’s why we use -30 in gamma at `addWeighted` Method
> 

# `Chinar_(P11)`

![4.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb40c4ed-dc7a-4827-b23b-0666bb967f51/4.jpg)

```python
image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Chinar_(P11)\\image_ ('+str(i+1)+').jpg', 0)
    median = cv2.medianBlur(image, 11)
    out = cv2.addWeighted(median, 3, median, 0, -70)

    (T, threshInv) = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\4-Chinar_(P11)\\'+str(i+1)+'.jpg', opening)
    img2 = cv2.imread('C:\\Users\\kimok\\Desktop\\saved\\4-Chinar_(P11)\\'+str(i+1)+'.jpg', 0)
    img3 = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_GroundTruth\\Chinar_(P11)\\image_ ('+str(i+1)+').JPG',0)
    # ==================
    img_np = np.array(img3)
    img2_np = np.array(img2)
    simxy = jaccard_binary(img_np, img2_np)
    sum_per=sum_per+simxy
    print("sum:"+str(sum_per))
totaAvg=(sum_per/103)*100
print("total percentage:"+str(totaAvg))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> Same as Above also here we need to make image more darker so we use -70 on gamma
> 

# Folders (5,6,7)⇒ `Jamun_(P5)`,`Jatropha_(P6)`,`Lemon_(P10)`

1. Method used 
a. `Jaccard` ⇒ to make comparison between images 

      b. `hisEqulColor`⇒ method that make equalization to colored image 

### Jaccard

```python
def jaccard_binary(x,y):
    interSec=np.logical_and(x,y)
    union =np.logical_or(x,y)
    similarity =interSec.sum() /float(union.sum())
    return similarity
```

### `hisEqulColor`

```python
defhisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
returnimg
```

> this Function take an image in BGR color format and convert it to `YCRCB`  why?
cause first : we need to make equalization to the image without damage the value of the color  we need representation format that only care about Intensity of the colors  when we separate them ,
For a grey-scale image, each pixel is represented by the intensity 
value (brightness); that is why we can feed the pixel values directly to
 the HE function. However, that is not how it works for an *RGB*-formatted color image. Each channel of the *R*, *G*, and *B* represents the intensity of the related color, not the intensity/brightness of the image as a whole. And so, **running HE on these color channels is NOT the proper way**.
> 

---

Comparison between RGB colors and YCbCr when we separate the colors 

---

# RGB
![rgb](https://user-images.githubusercontent.com/85907989/171633980-bcad4463-637d-4a94-9554-04601988d9c8.png)


## R
![R](https://user-images.githubusercontent.com/85907989/171634249-6cfdb072-d47d-4fbc-8781-ac5c583d7489.jpg)


## G

![G](https://user-images.githubusercontent.com/85907989/171634300-d1f0443a-ee48-4555-becc-f109102bf32f.jpg)


## B
![B](https://user-images.githubusercontent.com/85907989/171634346-c61f4679-04d6-41ff-8aba-792b2bb18b01.jpg)

# YCbCr

![ycbcr](https://user-images.githubusercontent.com/85907989/171634370-173c5506-edf7-4c84-8ec9-fd6a908dec3d.jpg)

## Y


![Y](https://user-images.githubusercontent.com/85907989/171634422-75a5128d-5e08-4741-848a-2d1ac59ab71a.jpg)

## CB
![Cb](https://user-images.githubusercontent.com/85907989/171634448-def9ce30-066f-4358-b893-d6809ba3e876.jpg)



## CR
![Cr](https://user-images.githubusercontent.com/85907989/171634480-3f1639f9-a1cf-4759-8178-120ea8d56dd4.jpg)


---

## 2- Code 
(`Jamun_(P5)`)

```python
image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Jamun_(P5)\\image_ ('+str(i+1)+').jpg')
    img2 = hisEqulColor(image)
    dst = cv2.equalizeHist(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    col=cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved_gray_histogram\\'+str(i+1)+'.jpg',img2)
    # ===================== median
    out = cv2.addWeighted(img2, 1, img2, 0, -50)
    gaussian_blur = cv2.GaussianBlur(src=out, ksize=(5, 5), sigmaX=5, sigmaY=5)
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharped_img = cv2.filter2D(gaussian_blur, -1, sharpen_filter)
    cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved\\'+str(i+1)+'.jpg', sharped_img)
    # ============================================== color mask HSV
    result = sharped_img.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for green color
    lower1 = np.array([35, 0, 0])
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
(T2, threshInv2) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    median2 = cv2.medianBlur(opening2, 11)
    ero2 = cv2.erode(median2, (10, 10), iterations=3)
    dilated = cv2.dilate(ero2.copy(), None, iterations=3)
    # ====================================================== Closing
    (T3, threshInv3) = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closing3 = cv2.morphologyEx(threshInv3, cv2.MORPH_CLOSE, kernel)![Uploading rgb.png…]()

```

> After I read the image send it to `hisEqul` method that equalize the image so all images be so similar to each other so when we perform any operation on them the values be the same for each images as can as possible 
(2)
then we `add weighted` method to change the gamma of the image to make it darker witch also decrease the white noise left in image   
(3 ) `GAussianBlur`   
method used to remove the noise in the image ‘ salt and pepper’  and also not making any blur like median so i can achieve good sharpen to image 
(4) Sharpen image ⇒ `filter2D` 
used to sharp the image using array List to move on all pixels 
(5) `Color threshold`   
used to focus only on a certain range of colors ‘ Green ’  i used the fill masked and initialize the lower and upper boundary 
(6) then i changed the image to gray level so i can change the number of channels so i can apply `OTSU threshold`   
     a. first i did the opening and closing threshold 
(7 ) then used `Median , erosion and dilated` ⇒ why ? first to remove the rest of the noise in the image , second to remove any pixels come out from the leaf and third just to fill any pixels that cleared cause of the erosion and create smooth edges as can as possible 
(8 ) `Closing  threshold =>`  
cause some leafs have a hole inside so i wanted to fill these holes
> 

# `Jatropha_(P6)`

```python
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
```

> Same as above just different values
> 

# `Lemon_(P10)`

```python
i in range(159):
    image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Lemon_(P10)\\image_ ('+str(i+1)+').jpg')
    img2 = hisEqulColor(image)
    dst = cv2.equalizeHist(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    col=cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved_gray_histogram\\'+str(i+1)+'.jpg',img2)
    out = cv2.addWeighted(img2, 1, img2, 0, -80)
    gaussian_blur = cv2.GaussianBlur(src=out, ksize=(3, 3), sigmaX=5, sigmaY=5)
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharped_img = cv2.filter2D(gaussian_blur, -1, sharpen_filter)
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved\\'+str(i+1)+'.jpg', sharped_img)
    result = sharped_img.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for green color
    lower1 = np.array([0, 0, 0])
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
    # cv2.imshow("gray",gray)
    # =============================================erosion
    ero = cv2.erode(gray, (10, 10), iterations=1)
    # ========================================================== opening and closing
    (T2, threshInv2) = cv2.threshold(ero, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    median2 = cv2.medianBlur(opening2, 15)
    ero2 = cv2.erode(median2, (10, 10), iterations=1)
    #=========================================================== Closing
    (T3, threshInv3) = cv2.threshold(ero2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closing3 = cv2.morphologyEx(threshInv3, cv2.MORPH_CLOSE, kernel)
```

> also Same as above 
just used an extra erosion method ⇒    `ero = cv2.erode(gray, (10, 10), iterations=1)`
cause the images was having white pixels around the leaf connected to it so i tried to remove like the edge rounded area of the leaf to reduce that noise 
and make it easer to median method later on to remove the rest of the noise outside the boarders of the leaf
> 

 

---

# folders (8,9,10)⇒ `Mango_(P0)` ,`Pomegranate_(P9)` ,`Pongamia_Pinnata_(P7)`

## first : method used in all of them

1. `jaccard_binary` ⇒ used to make the comparison  between images generated and the source one 
2. `Color threshold` ⇒ used to focus on limited colors in the image to better  results 
3. `erosion`  ⇒ used to get rid off the pixels that were connected to the leaf  
4. `gray`   ⇒ method used to convert the HSV colored image to gray so we can change the NO. of Channels to apply the `OTSU ( opening and closing ) threshold` 

## Second : method used only in `Pomegranate_(P9)`

## 1. first why we used it ?

## I fount that the folder have 150 image that really have a dark low brightness color

## and the rest of the images was very light color high contrast so i needed to to make balance between all the images so the code can work perfect in all of them .

## `hisEqulColor`

> method that take the BGR image and change its format to YCCBCR  to get more range of the intensity of the color so we can apply good equalization 
that first use :
> 

## `cv2.Split()`

## that split the YCbCR to three channels  y⇒ for brightness and the blue difference Cb and the red difference CR as we explained above

## `cv2.merge(channels,ycrcb)`

## which connect all the channels again after i perform the equalization

---

# second : Code

# 1. `Mango_(P0)`

```python
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
```

> after i made equalization and color threshold and change the color to gray to change the NO. of channels as i mentioned above 
now i will perform the erosion so i can remove the noise that is connected to the image 

then i will apply the `Opening and closing OTSU` 
and then i applied the `median` method so i can remove the rest of the noise in the image
> 

## 2. `Pomegranate_(P9)`

```python
image = cv2.imread('D:\\1.aast\\1.college\\term8\\Digital image processing\\sec -yousef\\_Output\\Pomegranate_(P9)\\image_ ('+str(i+1)+').jpg')
    img2 = hisEqulColor(image)
    median3=cv2.medianBlur(img2,3)
    median4=cv2.medianBlur(img2,7)
    dst = cv2.equalizeHist(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    col=cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(src1=median3, alpha=1,src2=median4,beta=0 ,gamma=-130)
    gaussian_blur = cv2.GaussianBlur(src=out, ksize=(3, 3), sigmaX=5, sigmaY=5)
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharped_img = cv2.filter2D(gaussian_blur, -1, sharpen_filter)
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved_gray_histogram\\'+str(i+1)+'.jpg', sharped_img)
    result = sharped_img.copy()
    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for green color
    lower1 = np.array([0, 0, 0])
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
    # =============================================erosion
    ero = cv2.erode(gray, (10, 10), iterations=1)
    # ========================================================== opening and closing
    (T2, threshInv2) = cv2.threshold(ero, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #========================================================= Removing the rest of the noise
    median2 = cv2.medianBlur(opening2, 11)
    ero2 = cv2.erode(median2, (10, 10), iterations=3)
    dilated = cv2.dilate(ero2.copy(), None, iterations=3)
    #======================================================= CLosing
```

![Screenshot 2022-05-24 022842.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/713c16f9-874d-4d23-9df0-6c0383e1f962/Screenshot_2022-05-24_022842.jpg)

```python
median3=cv2.medianBlur(img2,3)
median4=cv2.medianBlur(img2,7)
out = cv2.addWeighted(src1=median3, alpha=1,src2=median4,beta=0 ,gamma=-130)
```

# `Median 3`

# `Median 4`

![1.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/401913e7-e586-4523-8187-4914f5d34f7f/1.jpg)

> we used median 3 with low matrix value that’s why the image is not that clear but we want it like that cause we will use  addWieght later on
> 

![2.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eacbf9da-526c-4495-8cc9-d524738a4829/2.jpg)

### this one is much clear than the first cause we use 7*7 martix

# `addWeighted`

> after i read the file ⇒ apply median method on the image to remove the noise from it then 
i used `addWighted method =>` used 4 inputs `src1=median, alpha=3,src2=median,beta=0 ,gamma=-130`
this function work ⇒ choose 2 images and apply them together to get new enhanced one and the gamma to change the brightness of the image if the value was <0 then it will change to a darker scale if it was > 0 then it will change to a brighter image ….
> 

![5.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/483d7a02-e376-472b-b1eb-a043260140fa/5.jpg)

```python
sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharped_img = cv2.filter2D(gaussian_blur, -1, sharpen_filter)
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved_gray_histogram\\'+str(i+1)+'.jpg', sharped_img)
    result = sharped_img.copy()
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved\\'+str(i+1)+'.jpg', result)

    image1 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # lower boundary for green color
    lower1 = np.array([0, 0, 0])
    upper1 = np.array([65, 255, 255])
    # upper boundary for yellow color
    lower2 = np.array([66, 0, 0])
    upper2 = np.array([80, 255, 255])
    lower_mask = cv2.inRange(image1, lower1, upper1)
    upper_mask = cv2.inRange(image1, lower2, upper2)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(result, result, mask=full_mask)
```

> sharpen image method used to sharp the image by using a `2d filter` that takes the image generated by the 3*3 matrix and the center value which is changed to change the value of the sharpen image and after the filter2d method this image after applying the 3*3 matrix mask it on the original one to enhance it .
`cvtColor=>`   here i want to change the color mode from BGR to HSV cause it has much wider values of color ‘ intensity ’ and to perform good `Full color threshold Mask` ⇒ give it the lower and the upper boundary values to target them and combine them together in `full mask`

`cv2.bitwise_and`  ⇒ first it make a black and white matrix from the images I have then if the black matrix pixels with the white one i goes to the mask variable and fill it from the same pixel from the original image
> 

# `color threshold Mask`

![6.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fda73121-972a-4eb2-9fde-788101dc3138/6.jpg)

# `cv2.bitwise_and`

![bitwise_and.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c7cede34-7fb1-4d77-b2e6-9c974f81739d/bitwise_and.jpg)

---

```python
ero = cv2.erode(gray, (10, 10), iterations=1)
    # cv2.imwrite('C:\\Users\\kimok\\Desktop\\saved\\9-saved_gray_histogram\\'+str(i+1)+'.jpg', ero)
```

> as we can see there is noise around the leaf so i use erosion to remove it
> 

![ero.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f8c59ab6-e8dc-4f95-b26a-5804b9db9abc/ero.jpg)

```python
(T2, threshInv2) = cv2.threshold(ero, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(threshInv2, cv2.MORPH_CLOSE, kernel)
    opening2 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
```

> then i used the OTSU threshhold to get the optimal threshHold value doing opening and closing
> 

## `the image after opening`

![opening.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06da2f91-8858-4b05-a940-54eee7a5cfc2/opening.jpg)

> here we are most of noise is removed cause of the opening and closing operation we made but as we can see there exists noise around it so i will use `median` to remove it
> 

## `Median`

```python
median2 = cv2.medianBlur(opening2, 11)
```

![median2.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0dc16a64-e39e-4e16-909c-e1d209a07e07/median2.jpg)

## then i used `Closing` so if there is any holes in the image i will close it like the hole at the end of  the leeaf 
`Closing`

![1.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46132d0f-f11d-4e20-81a2-6100c88ee5c5/1.jpg)

## 3.`Pongamia_Pinnata_(P7)`

```python
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
```

> Same as above
> 

---
