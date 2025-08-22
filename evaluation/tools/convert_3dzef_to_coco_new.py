import os
import numpy as np
import json
import cv2

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '/data/data0/liuyiran/dataset/3dzef_joint/'
OUT_PATH = DATA_PATH + 'annotations_new/'
SPLITS = ['train']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


def xyxy2tlbr(box):
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]
    return box


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + (split if not HALF_VIDEO else 'train')
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'imgT'}, {'id': 2, 'name': 'imgF'}],
               'videos': []}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': seq})
            seq_path = '{}/{}/'.format(data_path, seq)
            # img_path = seq_path + 'img1/'
            path_list = ['imgT', 'imgF']
            for cate in path_list:
                ann_path = seq_path + 'gt/gt.txt'
                img_path = '{}/{}/'.format(seq_path, cate)
                image = os.listdir(img_path)
                num_images = len([image for image in image if 'jpg' in image])
                if HALF_VIDEO and ('half' in split):
                    image_range = [0, num_images // 2] if 'train' in split else \
                        [num_images // 2 + 1, num_images - 1]
                else:
                    image_range = [0, num_images - 1]
                for i in range(num_images):
                    if (i < image_range[0] or i > image_range[1]):
                        continue
                    image_info = {'file_name': '{}/{}/{:06d}.jpg'.format(seq, cate, i + 1),
                                  'id': image_cnt + i + 1,
                                  'frame_id': i + 1 - image_range[0],
                                  'prev_image_id': image_cnt + i if i > 0 else -1,
                                  'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                                  'video_id': video_cnt}
                    out['images'].append(image_info)
                print('{}: {} images'.format(seq, num_images))
                if split != 'test':
                    anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                    print(' {} ann images'.format(int(anns[:, 0].max())))
                    if cate == 'imgT':
                        for i in range(anns.shape[0]):
                            frame_id = int(anns[i][0])
                            if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                                continue
                            track_id = int(anns[i][1])
                            ann_cnt += 1
                            category_id = 1
                            ann = {'id': ann_cnt,
                                   'category_id': category_id,
                                   'image_id': image_cnt + frame_id,
                                   'track_id': track_id,
                                   'bbox': xyxy2tlbr(anns[i][7:11].tolist()),
                                   'space': anns[i][2:5].tolist(),
                                   'conf': float(1.0),
                                   'iscrowd': 0,
                                   'area': float(anns[i][9] * anns[i][10])
                                   }
                            out['annotations'].append(ann)
                    else:
                        for i in range(anns.shape[0]):
                            frame_id = int(anns[i][0])
                            if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                                continue
                            track_id = int(anns[i][1])
                            ann_cnt += 1
                            category_id = 2
                            ann = {'id': ann_cnt,
                                   'category_id': category_id,
                                   'image_id': image_cnt + frame_id,
                                   'track_id': track_id,
                                   'bbox': xyxy2tlbr(anns[i][14:18].tolist()),
                                   'space': anns[i][2:5].tolist(),
                                   'conf': float(1.0),
                                   'iscrowd': 0,
                                   'area': float(anns[i][16] * anns[i][17])
                                   }
                            out['annotations'].append(ann)
                image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))


