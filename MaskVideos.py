import cv2
from moviepy.editor import *
from moviepy.video.fx import *
from moviepy.video.tools.drawing import *
from moviepy.video.fx.all import *
from PIL import Image, ImageDraw
import gizeh
import numpy as np

def main():
    # Load video and detect frames (edges) using OpenCV
    # For demonstration, assume we have the frame coordinates
    frame_coordinates = [(100, 100), (200, 100), (200, 200), (100, 200)]  # Example coordinates

    video_path = 'images/video01.mp4'

    #mask = Image.new('L', [1024, 1024], 0)
    #draw = ImageDraw.Draw(mask)
    #draw.rectangle([(0, 100), (0, 100)], 255)
    #mask_array = np.array(mask)
    #im = Image.open("images/wall01.jpg")
    #im_array = np.array(im)

    video = VideoFileClip(video_path)
    clip = crop(video, x1=50, y1=50, x2=100, y2=100)
    clip.write_videofile("test2.mp4")
    return
    #mask_clip = ImageClip(im_array).set_duration(video.duration).set_opacity(0.5)
    size = video.size
    mask = color_split(size,
                       x=50, col1=0, col2=1)
    mask_clip = ImageClip(mask, ismask=True)

    clip_left = video.set_mask(mask_clip)
    clip_left.write_videofile("test.mp4")
    return
    masked_video = video.set_mask(mask_clip.to_mask())

    output_path = 'masked_output.mp4'
    print(output_path)
    masked_video.write_videofile("masked_output.mp4")


if __name__ == "__main__":
    main()
