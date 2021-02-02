import argparse
import cv2
import numpy as np
from PIL import Image
import pcl

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", action="store", dest="i", help="input .png file")
parser.add_argument("-o", "--output", action="store", dest="o", nargs='?', default="out.pcd", help="output .pcd file")

args = parser.parse_args()

cv_img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)

height, width = cv_img.shape
actual_height = height / 3

intensity_image = cv_img[0:int(actual_height), 0:width]
heightmap_raw = cv_img[int(actual_height):height, 0:width]

heightmap_raw_height, heightmap_raw_width = heightmap_raw.shape
size = heightmap_raw.shape[0] * (heightmap_raw.shape[1] / 2)
table = []

for y in range(int(heightmap_raw_height)):
        for x in range(int(heightmap_raw_width / 2)):
            table.append(heightmap_raw[y, x * 2] + 256 * heightmap_raw[y, x * 2 + 1])

correct_heightmap = np.reshape(table, (-1, int(heightmap_raw_width)))
correct_heightmap_height, correct_heightmap_width = correct_heightmap.shape

png_img = Image.open(args.i)

camera_z = float(png_img.text['CameraZ']) / 1000
ValidHeightmap = int(float(png_img.text['ValidHeightmap']))
ValidIntensity = int(float(png_img.text['ValidIntensity']))
x0 = float(png_img.text['X0']) / 1000
y0 = float(png_img.text['Y0']) / 1000
z0 = float(png_img.text['Z0']) / 1000
dx = float(png_img.text['Dx']) / 1000
dy = float(png_img.text['Dy']) / 1000
dz = float(png_img.text['Dz']) / 1000

points = np.uint16(correct_heightmap)
points_rows, points_cols = points.shape
cloud = np.zeros((points.size, 3), dtype=np.float32)

y_offset = points_rows * dy / 2
y0 -= y_offset
z_offset = 1.091

k = 0
for y in range(points_rows):
    for x in range(points_cols):
        dist = points[y][x]

        cloud[k][0] = np.float32(x0 + x * dx)
        cloud[k][1] = np.float32(y0 + y * dy)
        if correct_heightmap[y, x] == 0:
            cloud[k][2] = float('nan')
        else:
            cloud[k][2] = np.float32((z0 + z_offset) - (dist * dz))

        k = k + 1

pointcloud = pcl.PointCloud()
pointcloud.from_array(cloud)

pcl.save(pointcloud, args.o)
