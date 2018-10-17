import cv2
import glob
import argparse

parser = argparse.ArgumentParser(description='Find images with no faces that can detected in folder')
parser.add_argument('-dir_path')
parser.add_argument('-output_file_name')

args = parser.parse_args()

print(args.dir_path)

if(args.dir_path):
    img_list = glob.glob(args.dir_path + '/*.jpg')

    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    for img_name in img_list:
        img = cv2.imread(img_name)

        faces = haar_face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors = 5)

        try:
            img.shape[0], img.shape[1]
        except:
            with open(args.output_file_name + "_no_image_detected.txt", "a") as myfile:
                myfile.write(img_name + '\n')
