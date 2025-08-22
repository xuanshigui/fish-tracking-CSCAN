import numpy as np
import pandas as pd


def read_txt(txt_path):
    txt_dict = dict()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            img_id = linelist[0]
            obj_id = int(float(linelist[1]))
            bbox = [float(linelist[2]), float(linelist[3]),
                    float(linelist[2]) + float(linelist[4]),
                    float(linelist[3]) + float(linelist[5]), int(obj_id), float(linelist[6])]
            if int(img_id) in txt_dict:
                txt_dict[int(img_id)].append(bbox)
            else:
                txt_dict[int(img_id)] = list()
                txt_dict[int(img_id)].append(bbox)
    return txt_dict


def get_life_span(lifespan_dict, object_id):
    if not lifespan_dict.__contains__(object_id):
        lifespan_dict[object_id] = 1
    else:
        lifespan_dict[object_id] += 1
    return lifespan_dict[object_id]


def box_to_center(bbox):
    point_x = int((bbox[0] + bbox[2]) / 2)
    point_y = int((bbox[1] + bbox[3]) / 2)
    return point_x, point_y


if __name__ == '__main__':

    cam = 2
    theta = 0
    # seq_names = ['ZebraFish_01', 'ZebraFish_02', 'ZebraFish_03', 'ZebraFish_04']
    seq_names = ['ZebraFish-01', 'ZebraFish-02', 'ZebraFish-03', 'ZebraFish-04']
    top_view_path = '/ai/liuyiran/oc-sort/evaldata/trackers/mot_challenge/test/yolox_s_fish3d_front_results/data/'
    # front_view_path = '/ai/liuyiran/oc-sort/evaldata/trackers/mot_challenge/test/yolox_s_fish3d_front_results/data'
    result_root = '/ai/liuyiran/oc-sort/2DTo3D/front_view/'
    for seq_name in seq_names:
        # A .csv for a video
        csv_filename = result_root + seq_name + '.csv'
        columns = ['', 'frame', 'id', 'cam', 'x', 'y', 'tl_x', 'tl_y', 'c_x', 'c_y', 'w', 'h', 'theta', 'l_x',
                   'l_y', 'r_x', 'r_y', 'aa_tl_x', 'aa_tl_y', 'aa_w', 'aa_h']
        df = pd.DataFrame(columns=columns)
        top_view_txt = top_view_path + seq_name + '.txt'
        top_view_dict = read_txt(top_view_txt)
        # front_view_path = read_txt(front_view_path)
        life_span_dict = dict()
        for img_id in sorted(top_view_dict.keys()):

            for bbox in top_view_dict[img_id]:
                object_id = bbox[4]
                life_span = get_life_span(life_span_dict, object_id)
                center = box_to_center(bbox)
                row = [int(life_span), int(img_id), int(object_id), cam, center[0], center[1], bbox[0], bbox[1], center[0], center[1],
                       bbox[2], bbox[3], theta, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[1], bbox[0], bbox[1],
                       bbox[0]+bbox[2], bbox[1]+bbox[1]]
                df = df._append(pd.Series(row, index=columns), ignore_index=True)
        df.to_csv(csv_filename, index=False)
