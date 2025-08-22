from modules.comparison.MatchingOnline import TrackletMatcher
from modules.visualization.visualize_trajectory import read_txt
import pandas as pd
import os.path as osp
import os

class Track:
    def __init__(self, frame, track_id, pos):
        self.frame = frame
        self.id = track_id
        self.pos = pos


def write_3d_results(all_tracks, output_dir, seq_name=None):

    assert seq_name is not None, "[!] No seq_name, probably using combined database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, seq_name + '_3d.csv')

    data = []
    for values in all_tracks.values():
        for id, value in values.items():
            row = {'frame': id,
                   'id': value['id'],
                   'err': value['err'],
                   '3d_x': value['3d_x'],
                   '3d_y': value['3d_y'],
                   '3d_z': value['3d_z'],
                   'cam1_x': value['cam1_x'],
                   'cam1_y': value['cam1_y'],
                   'cam2_x': value['cam2_x'],
                   'cam2_y': value['cam2_y'],
                   'cam1_proj_x': -1.0,
                   'cam1_proj_y': -1.0,
                   'cam2_proj_x': -1.0,
                   'cam2_proj_y': -1.0,
                   'cam1_tl_x': value['cam1_tl_x'],
                   'cam1_tl_y': value['cam1_tl_y'],
                   'cam1_c_x': value['cam1_c_x'],
                   'cam1_c_y': value['cam1_c_y'],
                   'cam1_w': value['cam1_w'],
                   'cam1_h': value['cam1_h'],
                   'cam1_theta': 0.0,
                   'cam1_aa_tl_x': value['cam1_tl_x'],
                   'cam1_aa_tl_y': value['cam1_tl_y'],
                   'cam1_aa_w': value['cam1_w'],
                   'cam1_aa_h': value['cam1_h'],
                   'cam1_frame': id,
                   'cam2_tl_x': value['cam2_tl_x'],
                   'cam2_tl_y': value['cam2_tl_y'],
                   'cam2_c_x': value['cam2_c_x'],
                   'cam2_c_y': value['cam2_c_y'],
                   'cam2_w': value['cam2_w'],
                   'cam2_h': value['cam2_h'],
                   'cam2_theta': 0.0,
                   'cam2_aa_tl_x': value['cam2_tl_x'],
                   'cam2_aa_tl_y': value['cam2_tl_y'],
                   'cam2_aa_w': value['cam2_w'],
                   'cam2_aa_h': value['cam2_h'],
                   'cam2_frame': id}
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    df.to_csv(file)

def main():
    seq_names = ['ZebraFish_05', 'ZebraFish_06', 'ZebraFish_07', 'ZebraFish_08']
    # construct a Tracklet Matcher
    online_matcher = TrackletMatcher('/data/data0/liuyiran/Zef3d/eval_result/' + 'ZebraFish_05')
    method = 'ncaa39online'
    output_dir = '/data/data0/liuyiran/Zef3d/modules/comparison/tracking_3d_results/' + method + '/'
    for seq_name in seq_names:
        txt_path_top = '/data/data0/liuyiran/Zef3d/eval_result/fish3d_top_joint_ncaa39/' + seq_name + '.txt'
        txt_path_front = '/data/data0/liuyiran/Zef3d/eval_result/fish3d_front_joint_ncaa39/' + seq_name + '.txt'
        results_dict = dict()
        txt_dict_top = read_txt(txt_path_top)
        txt_dict_front = read_txt(txt_path_front)
        for img_id in sorted(txt_dict_front.keys()):
            if img_id in txt_dict_top.keys():
                top_objects = txt_dict_top[img_id]
            else:
                top_objects = None
            if img_id in txt_dict_front.keys():
                front_objects = txt_dict_front[img_id]
            else:
                front_objects = None
            # make object list
            top_objects_list = []
            if top_objects is not None:
                for bbox in top_objects:
                    # xyxy
                    pos = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    track = Track(img_id, bbox[4], pos)
                    top_objects_list.append(track)
            front_objects_list = []
            if front_objects is not None:
                for bbox in front_objects:
                    pos = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    track = Track(img_id, bbox[4], pos)
                    front_objects_list.append(track)
            # start to associate
            tracklet_list = online_matcher.get_3d_tracks(top_objects_list, front_objects_list)
            # save the result in a dict
            for t in tracklet_list:
                if t['id'] not in results_dict.keys():
                    results_dict[t['id']] = {}
                results_dict[t['id']][img_id] = t
            # write the results to a csv
            write_3d_results(results_dict, output_dir, seq_name)

if __name__ == '__main__':
    main()
