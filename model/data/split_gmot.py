import os
import numpy as np
import cv2

if __name__ == '__main__':

    DATA_PATH = '/data/data0/liuyiran/dataset/gmot40'
    OUT_PATH = '/data/data0/liuyiran/dataset/gmot_fish'
    seqs = os.listdir(DATA_PATH)
    split_ratio = 0.7
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0

    for seq in sorted(seqs):
        if '.DS_Store' in seq:
            continue
        if 'MOT17' in DATA_PATH and 'SDP' not in seq:
            continue
        video_cnt += 1
        seq_path = '{}/{}/'.format(DATA_PATH, seq)
        img_path = seq_path + 'img1/'
        ann_path = seq_path + 'gt/gt.txt'

        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        test_count = 0
        train_split = int(num_images * split_ratio)
        # 分割标注
        for index in range(0, len(anns)):
            if int(anns[index][0]) <= train_split:
                output_seq = '{}/{}/{}/'.format(OUT_PATH, 'train', seq)
                output_path = output_seq + 'gt/gt.txt'
                labels = '{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:f},{:d},{:d},{:d}\n'\
                    .format(int(anns[index][0]), int(anns[index][1]), anns[index][2], anns[index][3], anns[index][4],
                            anns[index][5], anns[index][6], int(anns[index][7]), int(anns[index][8]), int(anns[index][9]))
                with open(output_path, 'a') as f:
                    f.write(labels)
            else:
                output_seq = '{}/{}/{}/'.format(OUT_PATH, 'test', seq)
                output_path = output_seq + 'gt/gt.txt'
                labels = '{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:f},{:d},{:d},{:d}\n' \
                    .format(int(int(anns[index][0])-train_split-1), int(anns[index][1]), anns[index][2], anns[index][3],
                            anns[index][4], anns[index][5], anns[index][6], int(anns[index][7]), int(anns[index][8]),
                            int(anns[index][9]))
                with open(output_path, 'a') as f:
                    f.write(labels)
        # 分割图像
        for index in range(0, num_images):
            if index <= train_split:
                output_seq = '{}/{}/{}/'.format(OUT_PATH, 'train', seq)
                output_img = output_seq + 'img1/'
                img = cv2.imread(img_path + f'{index:06}.jpg')
                cv2.imwrite(output_img + f'{index:06}.jpg', img)
            else:
                output_seq = '{}/{}/{}/'.format(OUT_PATH, 'test', seq)
                output_img = output_seq + 'img1/'
                img = cv2.imread(img_path + f'{index:06}.jpg')
                cv2.imwrite(output_img + f'{int(index-train_split):06}.jpg', img)


