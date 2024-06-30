from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 80
pfov = 70

img = "images/Defisheye/Calibration_Image.jpg"
img_out = f"images/calibration_image_corrected_{dtype}_{format}_{pfov}_{fov}.jpg"
xcenter = -1

obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)

# To save image locally
obj.convert(outfile=img_out)

# To use the converted image in memory

new_image = obj.convert()