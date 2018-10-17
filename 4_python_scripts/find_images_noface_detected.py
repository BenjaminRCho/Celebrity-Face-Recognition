import cv2
import glob
import argparse

#set arg parse
parser = argparse.ArgumentParser(description='Find images with no faces that can detected in folder')
#directory path
parser.add_argument('-dir_path')
#file name
parser.add_argument('-output_file_name')
args = parser.parse_args()

print(args.dir_path)

if(args.dir_path):
    img_list = glob.glob(args.dir_path + '/*.jpg')

    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    for img_name in img_list:
        img = cv2.imread(img_name)

        faces = haar_face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors = 5)

        # if there are no faces found in image save to a textfile
        if len(faces) < 1:
            with open(args.output_file_name + '_no_faces_detected.txt', 'a') as myfile:
                myfile.write(img_name + '\n')
        else:
            continue
