import datetime

import cv2
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip
from moviepy.video.fx import *
from moviepy.video.tools.drawing import *
from moviepy.video.fx.all import *
import tkinter as tk
import gizeh
import numpy as np
import csv
import time
from defisheye import Defisheye

IsTest = True
Buffer = 0
AllowApproxFrameDetection = True
CALIBRATION_MARGIN = 10
CALIBRATION_MARGIN_WIDTH = 20
# Open the CSI camera using GStreamer pipeline
pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


def main():
    # Start projecter screen and capture coordinates of frames within space (edge detect)
    absoluteCanvasPos = GetCanvasSize()
    picFrames = FindFramesRelativePositions(absoluteCanvasPos)

    print(f'picFrames {picFrames}')

    print(f'canvas {absoluteCanvasPos}')

    videos = GetVideoPaths()
    totalClipLength = GetLongestVideoLength(videos)
    print(f'duration: {totalClipLength}')

    canvasHeight, canvasWidth = GetShapeHeightWidth(absoluteCanvasPos)
    print("canvasH" + str(canvasHeight))
    print("canvasW" + str(canvasWidth))

    # create background "mask"
    background_array = np.zeros((canvasHeight, canvasWidth, 3))
    black_image = ImageClip(background_array, duration=5)
    preprocessed_video_list = [black_image]
    # load videos for each frame

    for ii in range(0, picFrames.__len__()):
        video_path = videos[ii]
        testFrame = picFrames[ii]
        preprocessed_video_list.append(FitVideoToFrame(video_path, testFrame, totalClipLength))
        print(preprocessed_video_list[ii].duration)

    compositeVideo = CompositeVideoClip(preprocessed_video_list)
    compositeVideo.write_videofile("output_videos/test5_finished.mp4", audio=False, verbose=True)


def DisplayCalibrationSquare():
    window = tk.Tk()
    window.attributes('-fullscreen', True)
    window.update()

    projectionWidth = window.winfo_width()
    projectionHeight = window.winfo_height()

    print(f"width {projectionWidth} {projectionHeight}")

    canvas = tk.Canvas(window, width=projectionWidth, height=projectionHeight, bg='white')
    canvas.pack()

    margin = 20
    canvas.create_rectangle(CALIBRATION_MARGIN, CALIBRATION_MARGIN, projectionWidth-CALIBRATION_MARGIN, projectionHeight-CALIBRATION_MARGIN, fill='', outline='black', width=CALIBRATION_MARGIN_WIDTH)

    def ProcessAndCloseCalibrationSquare():
        CaptureWebcamPhoto(f"images/Calibration_Image.jpg")
        canvas.delete("all")
        DisplayWhiteSquare(projectionWidth, projectionHeight)

    def DisplayWhiteSquare(width, height):
        canvas.create_rectangle(CALIBRATION_MARGIN, CALIBRATION_MARGIN,
                                projectionWidth-CALIBRATION_MARGIN, projectionHeight-CALIBRATION_MARGIN,
                                fill='white', outline='black', width=2)
        window.after(3000, CaptureAndCloseWhiteSquare)

    def CaptureAndCloseWhiteSquare():
        CaptureWebcamPhoto(f"images/Calibration_Image_White.jpg")
        window.destroy()
        window.quit()

    window.after(3000, ProcessAndCloseCalibrationSquare)
    window.mainloop()
    window.quit()
    return projectionWidth, projectionHeight


def CaptureWebcamPhoto(outputFileName):

    if IsTest:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Could not open webcam.");
        exit()

    time.sleep(3)
    ret, frame = cap.read()

    if IsTest:
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(0)

    if ret:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = outputFileName
        #filename = f"images/Calibration_Image.jpg"
        cv2.imwrite(filename, frame)
    else:
        print("Error: Can't recieve frame. Exiting...")

    cap.release()
    return


def GetCanvasSize():
    if not IsTest:
        GetFishEyeCorrectionParameters()

        print("fov and pfov retrieved")
        CanvasTrueWidth, CanvasTrueHeight = DisplayCalibrationSquare()

    fov, pfov = CorrectFisheyeDistortion()
    print("corrected fisheye")
    projectionPlaneEdges = FindProjectionPlaneRelativeEdgesFromCam()
    print("canvas edges: " + str(projectionPlaneEdges))
    return projectionPlaneEdges
    #return [(0, 0), (800, 0), (800, 600), (0, 600)]

def GetFishEyeCorrectionParameters():
    window = tk.Tk()
    window.attributes('-fullscreen', True)
    window.update()

    projectionWidth = window.winfo_width()
    projectionHeight = window.winfo_height()

    CalibrationSquareSide = projectionHeight
    centerWidth = projectionWidth/2
    halfCalibrationSquareSide = CalibrationSquareSide/2

    leftSideCalibrationSquare = centerWidth-halfCalibrationSquareSide

    canvas = tk.Canvas(window, width=projectionWidth, height=projectionHeight, bg='white')
    canvas.pack()

    canvas.create_rectangle(leftSideCalibrationSquare, 0, leftSideCalibrationSquare + CalibrationSquareSide, CalibrationSquareSide, fill='#00FF00', outline='#00FF00')

    def ProcessAndCloseCalibrationSquare():
        CaptureWebcamPhoto(f"images/Calibration_Image_Green.jpg")
        canvas.delete("all")
        DisplayBlueSquare(leftSideCalibrationSquare, CalibrationSquareSide)

    def DisplayBlueSquare(leftSideCalibrationSquare, CalibrationSquareSide):
        canvas.create_rectangle(leftSideCalibrationSquare, 0,
                                leftSideCalibrationSquare + CalibrationSquareSide, CalibrationSquareSide,
                                fill='#0000FF', outline='#0000FF')
        window.after(3000, CaptureAndCloseBlueSquare)

    def CaptureAndCloseBlueSquare():
        CaptureWebcamPhoto(f"images/Calibration_Image_Blue.jpg")
        canvas.delete("all")
        DisplayRedSquare(leftSideCalibrationSquare, CalibrationSquareSide)

    def DisplayRedSquare(leftSideCalibrationSquare, CalibrationSquareSide):
        canvas.create_rectangle(leftSideCalibrationSquare, 0,
                                leftSideCalibrationSquare + CalibrationSquareSide, CalibrationSquareSide,
                                fill='#FF0000', outline='#FF0000')
        window.after(3000, CaptureAndCloseRedSquare)

    def CaptureAndCloseRedSquare():
        CaptureWebcamPhoto(f"images/Calibration_Image_Red.jpg")
        window.destroy()
        window.quit()

    window.after(3000, ProcessAndCloseCalibrationSquare)
    window.mainloop()
    window.quit()

    return

def CorrectFisheyeDistortion():
    image_path = f'images/Calibration_Image_Green.jpg'
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for green color
    lower_green = np.array([30, 45, 45])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('distorted_fisheye_mask', hsv)
    cv2.waitKey(0)


    dtype = 'linear'
    format = 'fullframe'
    fov = 80
    pfov = 70

    img_out = f"images/calibration_image_corrected_{dtype}_{format}_{pfov}_{fov}.jpg"
    xcenter = -1

    obj = Defisheye(image_path, dtype=dtype, format=format, fov=fov, pfov=pfov)

    # To save image locally
    obj.convert(outfile=img_out)

    return fov, pfov


def FindProjectionPlaneRelativeEdgesFromCam():
    if IsTest:
        image = cv2.imread('images/calibration_image_corrected_linear_fullframe_70_80.jpg')
    else:
        image = cv2.imread('images/Calibration_Image.jpg')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = np.array([30, 40, 35])
    upper_green = np.array([90, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    blurred_image = cv2.GaussianBlur(mask, (3, 3), 0)

    # edges is a 2d array (representing the image pixels)
    # values are at where edges are detected.
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=75)  # 100, 150
    contours, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    if AllowApproxFrameDetection:
        min_area = 30000
        max_area = 1000000
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if LooksLikeFrame(approx, min_area, max_area):
                if len(shapes) > 0:
                    shapes[0] = FindInnerMostSquare(approx, shapes[0])
                    # we have multiple matches, find the inner most one.

                shapes.append(approx)
                print("looks like frame!")
    else:
        for i, (cnt, hinfo) in enumerate(zip(contours, h[0])):
            ## hinfo[2] is children, [3] is parent. make sure there is a parent and no children here
            if hinfo[2] == -1 and hinfo[3] != -1:
                if cv2.contourArea(cnt) < cv2.arcLength(cnt, True):
                    continue
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 5:
                    print("pentagon")
                    continue
                elif len(approx) == 3:
                    print("triangle")
                elif len(approx) == 6:
                    print("hexagon")
                    continue
                elif len(approx) == 4:
                    print("square")
                    shapes.append(approx)
                elif len(approx) == 9:
                    print("half-circle")
                elif len(approx) > 15:
                    print("circle")
    if IsTest:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.imshow('edges', edges)
        cv2.waitKey(0)
        contour_image1 = image.copy()
        cv2.drawContours(contour_image1, contours, -1, (0, 0, 255), 2)  # Draw contours in red
        cv2.imshow('contour_image1', contour_image1)
        cv2.waitKey(0)
        #output_contour_path = 'images/TestOutput/contours_test.png'
        #cv2.imwrite(output_contour_path, contour_image1)

        contour_image2 = image.copy()
        for shape in shapes:
            cv2.drawContours(contour_image2, shape, -1, (0, 255, 0), 2)
        cv2.imshow('contour_image2', contour_image2)
        cv2.waitKey(0)
        #cv2.imwrite('images/TestOutput/shape_overlay_test.png', contour_image2)

    # Convert to list of tuples
    converted_coords = [(int(point[0][0]), int(point[0][1])) for point in shapes[0]]

    return converted_coords

def SimplifyImageDimensions(image, absoluteCanvasPos):
    src_pts = np.array(absoluteCanvasPos, dtype=np.float32)
    src_pts = src_pts[np.lexsort((src_pts[:, 1], src_pts[:, 0]))]
    width = max(np.linalg.norm(src_pts[2] - src_pts[0]), np.linalg.norm(src_pts[3] - src_pts[1]))
    height = max(np.linalg.norm(src_pts[1] - src_pts[0]), np.linalg.norm(src_pts[3] - src_pts[2]))

    sideLength = max(width, height)

    dst_pts = np.array([[0, 0], [0, sideLength - 1], [sideLength - 1, 0], [sideLength - 1, sideLength - 1]], dtype="float32")
    print("dest points: " + str(dst_pts) + " source points: " + str(src_pts))
    print("width:" + str(width) + " height:" + str(height))
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned_image = cv2.warpPerspective(image, M, (int(sideLength), int(sideLength)))

    cv2.imshow('alignedImage', aligned_image)
    cv2.waitKey(0)
    return aligned_image


def FindFrameAbsoluteEdgesFromCam(absoluteCanvasPos):
    if IsTest:
        image = cv2.imread('images/calibration_image_corrected_linear_fullframe_70_80.jpg')
    else:
        image = cv2.imread('images/Calibration_Image.jpg')

    alignedImage = SimplifyImageDimensions(image, absoluteCanvasPos)
    heights = [point[1] for point in absoluteCanvasPos]
    widths = [point[0] for point in absoluteCanvasPos]

    startY = min(heights) + Buffer
    endY = max(heights) - Buffer
    startX = min(widths) + Buffer
    endX = max(widths) - 2*Buffer
    cropped_image = image[startY:endY, startX:endX]
    hsv = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = np.array([30, 45, 25])
    upper_green = np.array([90, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)


    # sobel
    #sobelx = cv2.Sobel(alignedImage, cv2.CV_64F, 1, 0, ksize=5)
    #sobely = cv2.Sobel(alignedImage, cv2.CV_64F, 0, 1, ksize=5)
    #sobel_combined = cv2.sqrt(sobelx ** 2 + sobely ** 2)
    #sobel_combined = cv2.convertScaleAbs(sobel_combined)
    #cv2.imshow('sobel', sobel_combined)
    #cv2.waitKey(0)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(alignedImage, alignedImage, mask=mask_inv)
    img_gray = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2GRAY)
    # Apply Histogram Equalization
    # equalized = cv2.equalizeHist(img_gray)
    # Increase brightness and contrast
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 80  # Brightness control (0-100)
    bright_contrast = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    blurred_image = cv2.GaussianBlur(bright_contrast, (5, 5), 0)
    ret, img_inv_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    # edges is a 2d array (representing the image pixels)
    # values are at where edges are detected.
    edges = cv2.Canny(blurred_image, threshold1=20, threshold2=25) #100, 150
    #coordinates = np.column_stack(np.where(edges > 0))

    contours, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if IsTest:
        cv2.imshow('Shapes', edges)
        cv2.waitKey(0)
        cv2.imshow('photo cropped', bright_contrast)
        cv2.waitKey(0)
        contour_image = alignedImage.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # Draw contours in red
        #output_contour_path = 'images/TestOutput/contours_test.png'
        #cv2.imwrite(output_contour_path, contour_image)
        cv2.imshow('contour_image1', contour_image)
        cv2.waitKey(0)

    framesAbsoluteEdges = []
    if AllowApproxFrameDetection:
        min_area = 500
        max_area = 100000
        for cnt in contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if LooksLikeFrame(approx, min_area, max_area):
                if not IsFrameDuplicate(approx, framesAbsoluteEdges, endY-startY, endX-startX):
                    framesAbsoluteEdges.append(approx)
                    print("looks like frame!")
    else:
        for i, (cnt, hinfo) in enumerate(zip(contours, h[0])):
            ## hinfo[2] is children, [3] is parent. make sure there is a parent and no children here
            if hinfo[2] == -1 and hinfo[3] != -1:
                if cv2.contourArea(cnt) < cv2.arcLength(cnt, True):
                    continue
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 5:
                    print("pentagon")
                    continue
                elif len(approx) == 3:
                    print("triangle")
                elif len(approx) == 6:
                    print("hexagon")
                    continue
                elif len(approx) == 4:
                    print("square")
                    framesAbsoluteEdges.append(approx)
                elif len(approx) == 9:
                    print("half-circle")
                elif len(approx) > 15:
                    print("circle")
    if IsTest:
        contour_image = alignedImage.copy()
        for frame in framesAbsoluteEdges:
            cv2.drawContours(contour_image, frame, -1, (0, 255, 0), 2)
        cv2.imshow('contour_image1', contour_image)
        cv2.waitKey(0)

    return framesAbsoluteEdges

def LooksLikeFrame(approx, min_area=50, max_area=100000):
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        area = cv2.contourArea(approx)
        if min_area <= area <= max_area:
            return True
    return False

def IsFrameDuplicate(approx, framesAbsoluteEdges, imageHeight, imageWidth):
    maskNewShape = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
    cv2.drawContours(maskNewShape, [approx], -1, 255, thickness=cv2.FILLED)
    for frame in framesAbsoluteEdges:
        maskExistingShape = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
        cv2.drawContours(maskExistingShape, [frame], -1, 255, thickness=cv2.FILLED)
        intersection = cv2.bitwise_and(maskNewShape, maskExistingShape)
        intersection_area = cv2.countNonZero(intersection)
        if intersection_area > 0:
            print("duplicate frame")
            return True
    return False


def FindInnerMostSquare(newShape, oldShape):
    (x1, y1, w1, h1) = cv2.boundingRect(newShape)
    (x2, y2, w2, h2) = cv2.boundingRect(oldShape)

    if w1 < w2:
        print("using new shape!")
        return newShape
    else:
        print("using old shape!")
        return oldShape

def FindFramesRelativePositions(absoluteCanvasPos):
    shapes = FindFrameAbsoluteEdgesFromCam(absoluteCanvasPos)

    ## shapes are ordered liked this
    ## [(x0, y0), (x1, y1), (x2, y2), (x3. y3)]
    ##   *(x0=0,y0=0)             *(x1=x, y1=0)
    ##
    ##   *(x3=0, y3=y)            *(x2=x,y2=y)
    newShapes = []
    print(f'shapes {shapes}')
    for shape in shapes:
        reformattedShapeUnOrdered = []
        for pt in shape:
            reformatted_point = (pt[0][0] + CALIBRATION_MARGIN, pt[0][1] + CALIBRATION_MARGIN)
            reformattedShapeUnOrdered.append(reformatted_point)
        reformattedShape = []
        top_points = sorted(reformattedShapeUnOrdered, key=lambda p: (p[1], [0]))[:2]
        print(f'top points {top_points}')
        bottom_points = sorted(reformattedShapeUnOrdered, key=lambda p: (p[1], [0]))[2:]

        # For top points, sort by x to find left and right
        top_left = min(top_points, key=lambda p: p[0])
        print(f'top left: {top_left}')
        top_right = max(top_points, key=lambda p: p[0])

        # For bottom points, sort by x to find left and right
        bottom_left = min(bottom_points, key=lambda p: p[0])
        bottom_right = max(bottom_points, key=lambda p: p[0])
        reformattedShape.append(top_left)
        reformattedShape.append(top_right)
        reformattedShape.append(bottom_right)
        reformattedShape.append(bottom_left)

        newShapes.append(reformattedShape)

    return newShapes

def GetShapeHeightWidth(shapeCoordinates):
    heights = [point[1] for point in shapeCoordinates]
    widths = [point[0] for point in shapeCoordinates]

    frameHeight = max(heights) - min(heights)
    frameWidth = max(widths) - min(widths)

    return frameHeight, frameWidth

def GetVideoPaths():
    return ['input_videos/video01.mp4', 'input_videos/video02.mp4']

def FitVideoToFrame(video_path, frame, duration):
    frameHeight, frameWidth = GetShapeHeightWidth(frame)

    video_temp = VideoFileClip(video_path)
    videoWidth, videoHeight = video_temp.size
    print(f"videoWidth {videoWidth} videoHeight {videoHeight}")
    bindVideoSizeToHeight = True
    videoWidthframeWidthDiff = frameWidth - videoWidth
    videoHeightFrameHeighDiff = frameHeight - videoHeight

    if videoHeightFrameHeighDiff < videoWidthframeWidthDiff:
        bindVideoSizeToHeight = False

    videoWidthHeightRatio = videoWidth / videoHeight

    if bindVideoSizeToHeight:
        videoHeight = frameHeight
        videoWidth = videoHeight * videoWidthHeightRatio
    else:
        videoWidth = frameWidth
        videoHeight = videoWidth * videoWidthHeightRatio
    print(f"Final videoWidth {videoWidth} videoHeight {videoHeight} frameHeight {frameHeight}")
    return VideoFileClip(video_path) \
        .set_position(frame[0]) \
        .resize((videoWidth, videoHeight)) \
        .crop(x1=0, y1=0, x2=frameWidth, y2=frameHeight) \
        .loop(duration=duration)

def GetLongestVideoLength(videos):
    duration = 0;
    for video_path in videos:
        video_temp = VideoFileClip(video_path)
        if(video_temp.duration > duration):
            duration = video_temp.duration

    return duration

if __name__ == "__main__":
    main()
