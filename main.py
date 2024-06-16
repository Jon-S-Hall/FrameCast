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

IsTest = True
# Open the CSI camera using GStreamer pipeline
pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


def main():
    # Start projecter screen and capture coordinates of frames within space (edge detect)

    totalCanvasSize = GetCanvasSize()
    picFrames = GetPicFrames()

    print(f'picFrames {picFrames}')

    print(f'canvas {totalCanvasSize}')

    videos = GetVideoPaths()
    totalClipLength = GetLongestVideoLength(videos)
    print(f'duration: {totalClipLength}')

    canvasHeight, canvasWidth = GetShapeHeightWidth(totalCanvasSize)

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
    compositeVideo.write_videofile("output_videos/test5_finished.mp4")

def GetLongestVideoLength(videos):
    duration = 0;
    for video_path in videos:
        video_temp = VideoFileClip(video_path)
        if(video_temp.duration > duration):
            duration = video_temp.duration

    return duration

def DisplayCalibrationSquare():
    window = tk.Tk()
    window.attributes('-fullscreen', True)
    window.update()

    width = window.winfo_width()
    height = window.winfo_height()

    print(f"width {width} {height}")

    canvas = tk.Canvas(window, width=width, height=height, bg='white')
    canvas.pack()

    margin = 20
    canvas.create_rectangle(margin, margin, width-margin, height-margin, fill='', outline='black', width=10)
    canvas.create_rectangle(300, 350, 400, 500, fill='', outline='black', width=10)

    def ProcessAndCloseCalibrationSquare():
        CaptureWebcamPhoto()
        window.destroy()
        window.quit()

    window.after(3000, ProcessAndCloseCalibrationSquare)
    window.mainloop()
    window.quit()

def CaptureWebcamPhoto():

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
        filename = f"images/Calibration_Image.jpg"
        cv2.imwrite(filename, frame)
    else:
        print("Error: Can't recieve frame. Exiting...")


    cap.release()


def GetCanvasSize():
    # dummy return as of now
    DisplayCalibrationSquare()
    return [(0, 0), (800, 0), (800, 600), (0, 600)]

def GetWebcamPhotoEdges():
    if IsTest:
        image = cv2.imread('images/wall03.png')
    else:
        image = cv2.imread('images/Calibration_Image.jpg')


    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    ret, img_inv_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    # edges is a 2d array (representing the image pixels)
    # values are at where edges are detected.
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=150)
    #coordinates = np.column_stack(np.where(edges > 0))
    #cv2.imshow('Shapes', edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    contours, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.zeros_like(image)
    shapes = []
    for i, (cnt, hinfo) in enumerate(zip(contours, h[0])):
        ## hinfo[2] is children, [3] is parent. make sure there is a parent and no children here
        if hinfo[2] == -1 and hinfo[3] != -1:
            if cv2.contourArea(cnt) < cv2.arcLength(cnt, True):
                continue
            epsilon = 0.01 * cv2.arcLength(cnt, True)
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

    return shapes


def GetPicFrames():
    shapes = GetWebcamPhotoEdges()
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
            reformatted_point = (pt[0][0], pt[0][1])
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

if __name__ == "__main__":
    main()
