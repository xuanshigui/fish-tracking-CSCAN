import cv2
import numpy as np

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def read_txt(txt_path):
    txt_dict = dict()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            img_id = linelist[0]
            obj_id = linelist[1]
            # bbox = [float(linelist[14]), float(linelist[15]),
            #         float(linelist[14]) + float(linelist[16]),
            #         float(linelist[15]) + float(linelist[17]), float(linelist[12]), float(linelist[13]), int(obj_id)]
            bbox = [float(linelist[2]), float(linelist[3]),
                    float(linelist[2]) + float(linelist[4]),
                    float(linelist[3]) + float(linelist[5]), int(float(obj_id))]
            if int(img_id) in txt_dict:
                txt_dict[int(img_id)].append(bbox)
            else:
                txt_dict[int(img_id)] = list()
                txt_dict[int(img_id)].append(bbox)
    return txt_dict


def bbox_to_center(bbox):
    point_x = int((bbox[0] + bbox[2]) / 2)
    point_y = int((bbox[1] + bbox[3]) / 2)
    return [point_x, point_y]


def main():
    show_video_name = 'ZebraFish-06'
    show_video_name_ = 'ZebraFish_06'
    txt_path = '/data/data0/liuyiran/Zef3d/Naive/top_results_58/' + show_video_name_ + '.txt'
    txt_dict = read_txt(txt_path)
    img_path = '/data/data0/liuyiran/Zef3d/3DZeF20/test/' + show_video_name + '/cam1/'
    color_list = colormap()
    for img_id in sorted(txt_dict.keys()):
        img_name = img_path + '{:0>6d}.jpg'.format(img_id)
        ori_img = cv2.imread(img_name)
        for bbox in txt_dict[img_id]:
            # pick out object
            cv2.rectangle(ori_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          color_list[bbox[4] % 79].tolist(), thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(ori_img, '{}'.format(bbox[4]), (int(bbox[2]), int(bbox[1])), font, 1.5,
                        color_list[bbox[4] % 79].tolist(), 4)
            # cv2.circle(ori_img, (int(bbox[4]), int(bbox[5])), 1, color_list[bbox[6] % 79].tolist(),
            #            thickness=8)
        cv2.imwrite(f"/data/data0/liuyiran/Zef3d/trajectory/naive/{show_video_name_}/{img_id:06}_traj.jpg", ori_img)
    print(show_video_name, "Done")


if __name__ == '__main__':
    main()