import numpy as np
import cv2
import slide

def save_image(image_mat, image_name):

    cv2.imwrite(image_name, image_mat)
    return

def image_segmentation(hsv_img):
    
    # Taking slices of the individual HSV components
    h_slice = hsv_img[:,:,0]
    s_slice = hsv_img[:,:,1]
    v_slice = hsv_img[:,:,2]

    # Taking Image Gradient using Sobel operator in x axis
    sobel_x = cv2.Sobel(v_slice, cv2.CV_64F, 1, 0)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(255*sobel_x/np.max(sobel_x))

    # Binarisation of Image gradient
    sobel_x_binary = np.zeros_like(sobel_x)
    sobel_x_binary[(sobel_x >= 15) & (sobel_x <= 255)] = 1

    # Binarisation of Saturation component of image
    s_slice_binary = np.zeros_like(s_slice)
    s_slice_binary[(s_slice >= 100) & (s_slice <= 255)] = 1

    # Combining gradient and saturation binary images using OR
    combined_binary = np.zeros_like(sobel_x_binary)
    combined_binary[(s_slice_binary == 1) | (sobel_x_binary == 1)] = 1

    return combined_binary

def roi_warp(img,
             src=np.float32([(0.37,0.45),(0.8,0.45),(0.24,0.75),(1,0.75)]),
             dst=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
             dst_size=(1640,590)):
    x_lim = img.shape[1]
    y_lim = img.shape[0]
    img_sz = [(x_lim, y_lim)]
    img_sz = np.float32(img_sz)
    src = src* img_sz
    image = img
    dst_sz = np.float32(dst_size)
    dst = dst * dst_sz
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, P, dst_size)
    
    # return warped image
    return warped

def warp_inv(img,
            src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
            dst=np.float32([(0.37,0.69),(0.8,0.69),(0.1,1),(1,1)]),
            dst_size=(1640,590)):
    
    x_lim = img.shape[1]
    y_lim = img.shape[0]
    img_sz = [(x_lim, y_lim)]
    img_sz = np.float32(img_sz)
    src = src* img_sz
    dst_sz = np.float32(dst_size)
    dst = dst * dst_sz
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, P, dst_size)
    
    # return warped image
    return warped

def draw_lanes(img, left_fit, right_fit):
    img_shp = img.shape[0]-1
    ploty = np.linspace(0, img_shp, img_shp+1)
    color_img = np.zeros_like(img)
    left = np.vstack([left_fit, ploty])
    left = [np.transpose(left)]
    left =  np.array(left)
    right = np.vstack([right_fit, ploty])
    right = np.transpose(right)
    right = [np.flipud(right)]
    right = np.array(right)

    points = np.hstack((left, right))
    pts = np.int_(points)
    cv2.fillPoly(color_img, pts, (0,200,255))
    wt = 0.7
    inv_perspective = warp_inv(color_img)
    param = 0.0
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, wt, int(param))

    return inv_perspective

def find_lane(image_path):

    # Converting BGR image to HSV colorspace
    image = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Performing image segmentation to accentuate lanes
    bin_image = image_segmentation(hsv_img)

    # Selecting ROI and performing warping
    mask_img = roi_warp(bin_image)

    # Applying sliding window algorithm
    window_img, curves, lanes, ploty = slide.sliding_window(mask_img)

    # Drawing applied lanes on original image
    lane_img = draw_lanes(image, curves[0],curves[1])

    return lane_img