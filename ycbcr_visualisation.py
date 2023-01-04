import cv2
import numpy as np

img = cv2.imread(r"content_img/boten_field.jpg")

imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

Y,Cr,Cb = cv2.split(imgYCC)

half = np.array([[127]*Y.shape[1]]*Y.shape[0]).astype(Y.dtype)

merge_Y = cv2.merge([Y, half, half])
merge_Cb = cv2.merge([half, half, Cb])
merge_Cr = cv2.merge([half, Cr, half])

merge_Y = cv2.cvtColor(merge_Y, cv2.COLOR_YCrCb2BGR)
merge_Cb = cv2.cvtColor(merge_Cb, cv2.COLOR_YCrCb2BGR)
merge_Cr = cv2.cvtColor(merge_Cr, cv2.COLOR_YCrCb2BGR)

cv2.imwrite(r'output_frames/ycrcb_vis/Y.png', merge_Y)
cv2.imwrite(r'output_frames/ycrcb_vis/Cb.png', merge_Cb)
cv2.imwrite(r'output_frames/ycrcb_vis/Cr.png', merge_Cr)