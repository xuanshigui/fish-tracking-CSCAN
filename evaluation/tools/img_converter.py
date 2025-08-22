import os
import cv2


def main():
    path = '/data/data0/liuyiran/dataset/3dzef_sap/imgT/train/ZebraFish_08/img1/'
    file_list = os.listdir(path)
    for file_name in file_list:
        img = cv2.imread(path+file_name)
        new_name = file_name.replace('PNG', 'jpg')
        cv2.imwrite(path+new_name, img)


if __name__ == '__main__':
    main()