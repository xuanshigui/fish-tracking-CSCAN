import time

import pandas as pd
import warnings
import argparse
from tools.reconstruction import read_txt
from tools.reconstruction import box_to_center
from tools.reconstruction import get_life_span
from modules.reconstruction.TrackletMatching import TrackletMatcher
from modules.reconstruction.TrackletMatching import combine2DTracklets, getDropIndecies
from modules.evaluation.MOT_evaluation import MOT_Evaluation
import networkx as nx
import numpy as np
import os


warnings.filterwarnings('ignore')


def txt2dataframe(seq_name, cam, txt_path):
    theta = 0
    # csv_filename = result_root + seq_name + '.csv'
    columns = ['', 'frame', 'id', 'cam', 'x', 'y', 'tl_x', 'tl_y', 'c_x', 'c_y', 'w', 'h', 'theta', 'l_x',
               'l_y', 'r_x', 'r_y', 'aa_tl_x', 'aa_tl_y', 'aa_w', 'aa_h']
    df = pd.DataFrame(columns=columns)
    txt_file = txt_path + seq_name + '.txt'
    view_dict = read_txt(txt_file)
    # front_view_path = read_txt(front_view_path)
    life_span_dict = dict()
    for img_id in sorted(view_dict.keys()):
        for bbox in view_dict[img_id]:
            object_id = bbox[4]
            life_span = get_life_span(life_span_dict, object_id)
            center = box_to_center(bbox)
            row = [int(life_span), int(img_id), int(object_id), cam[-1], center[0], center[1], bbox[0], bbox[1],
                   center[0], center[1],
                   bbox[2], bbox[3], theta, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[1], bbox[0],
                   bbox[1],
                   bbox[0] + bbox[2], bbox[1] + bbox[1]]
            df = df._append(pd.Series(row, index=columns), ignore_index=True)
    # df.to_csv(csv_filename, index=False)
    return df


def tracklet_match(tm,):
    global path, df
    tm.createNodes()
    tm.connectNodes3D()
    csv = pd.DataFrame()
    mergedCount = 0
    while (True):
        if (len(tm.graph.nodes) == 0):
            break
        # # Find the largest path through the graph
        path = nx.dag_longest_path(tm.graph)

        allFrames = []
        for p in path:
            allFrames += list(tm.triangulated[p].frame)
        toBeRemoved = []
        print("Best path:")
        for p in path:
            print(" ", p)
            # Save triangulated 3D information to CSV
            track3d = tm.triangulated[p]

            df = pd.DataFrame({
                'frame': track3d.frame,
                'id': [mergedCount] * len(track3d.frame),
                'err': track3d.errors,
                '3d_x': [q[0] for q in track3d.positions3d],
                '3d_y': [q[1] for q in track3d.positions3d],
                '3d_z': [q[2] for q in track3d.positions3d],
                'cam1_x': [q[0] for q in track3d.cam1positions],
                'cam1_y': [q[1] for q in track3d.cam1positions],
                'cam2_x': [q[0] for q in track3d.cam2positions],
                'cam2_y': [q[1] for q in track3d.cam2positions],
                'cam1_proj_x': [q[0] for q in track3d.cam1reprojections],
                'cam1_proj_y': [q[1] for q in track3d.cam1reprojections],
                'cam2_proj_x': [q[0] for q in track3d.cam2reprojections],
                'cam2_proj_y': [q[1] for q in track3d.cam2reprojections],
                'cam1_tl_x': [q[0] for q in track3d.cam1bbox],
                'cam1_tl_y': [q[1] for q in track3d.cam1bbox],
                'cam1_c_x': [q[2] for q in track3d.cam1bbox],
                'cam1_c_y': [q[3] for q in track3d.cam1bbox],
                'cam1_w': [q[4] for q in track3d.cam1bbox],
                'cam1_h': [q[5] for q in track3d.cam1bbox],
                'cam1_theta': [q[6] for q in track3d.cam1bbox],
                'cam1_aa_tl_x': [q[7] for q in track3d.cam1bbox],
                'cam1_aa_tl_y': [q[8] for q in track3d.cam1bbox],
                'cam1_aa_w': [q[9] for q in track3d.cam1bbox],
                'cam1_aa_h': [q[10] for q in track3d.cam1bbox],
                'cam1_frame': track3d.cam1frame,
                'cam2_tl_x': [q[0] for q in track3d.cam2bbox],
                'cam2_tl_y': [q[1] for q in track3d.cam2bbox],
                'cam2_c_x': [q[2] for q in track3d.cam2bbox],
                'cam2_c_y': [q[3] for q in track3d.cam2bbox],
                'cam2_w': [q[4] for q in track3d.cam2bbox],
                'cam2_h': [q[5] for q in track3d.cam2bbox],
                'cam2_theta': [q[6] for q in track3d.cam2bbox],
                'cam2_aa_tl_x': [q[7] for q in track3d.cam2bbox],
                'cam2_aa_tl_y': [q[8] for q in track3d.cam2bbox],
                'cam2_aa_w': [q[9] for q in track3d.cam2bbox],
                'cam2_aa_h': [q[10] for q in track3d.cam2bbox],
                'cam2_frame': track3d.cam2frame})

            # Save information from parent tracks which are
            # not already present in the saved 3D track
            for parent in [track3d.cam1Parent, track3d.cam2Parent]:
                for f in parent.frame:
                    if (f in allFrames):
                        continue

                    newRow = pd.DataFrame({
                        'frame': [f],
                        'id': [mergedCount],
                        'err': [-1],
                        '3d_x': [-1],
                        '3d_y': [-1],
                        '3d_z': [-1],
                        'cam1_x': [-1],
                        'cam1_y': [-1],
                        'cam2_x': [-1],
                        'cam2_y': [-1],
                        'cam1_proj_x': [-1.0],
                        'cam1_proj_y': [-1.0],
                        'cam2_proj_x': [-1.0],
                        'cam2_proj_y': [-1.0],
                        'cam1_tl_x': [-1.0],
                        'cam1_tl_y': [-1.0],
                        'cam1_c_x': [-1.0],
                        'cam1_c_y': [-1.0],
                        'cam1_w': [-1.0],
                        'cam1_h': [-1.0],
                        'cam1_theta': [-1.0],
                        'cam1_aa_tl_x': [-1.0],
                        'cam1_aa_tl_y': [-1.0],
                        'cam1_aa_w': [-1.0],
                        'cam1_aa_h': [-1.0],
                        'cam1_frame': [-1],
                        'cam2_tl_x': [-1.0],
                        'cam2_tl_y': [-1.0],
                        'cam2_c_x': [-1.0],
                        'cam2_c_y': [-1.0],
                        'cam2_w': [-1.0],
                        'cam2_h': [-1.0],
                        'cam2_theta': [-1.0],
                        'cam2_aa_tl_x': [-1.0],
                        'cam2_aa_tl_y': [-1.0],
                        'cam2_aa_w': [-1.0],
                        'cam2_aa_h': [-1.0],
                        'cam2_frame': [-1]})

                    # Update cam2 with correct 2D positions
                    pointType = "kpt"
                    if parent.cam == 2 and not tm.camera2_useHead:
                        maxTemporalDiff = 10
                        indToPoint = {0: "l", 1: "c", 2: "r"}
                        track3DFrames = np.asarray(track3d.frame)
                        cam2Positions = np.asarray(track3d.cam2positions)

                        frameDiff = track3DFrames - f
                        validFrames = track3DFrames[np.abs(frameDiff) <= maxTemporalDiff]

                        hist = np.zeros((3))
                        for f_t in validFrames:
                            ftPoint = np.asarray(cam2Positions[track3DFrames == f_t])
                            points = np.zeros((3))
                            points[0] = np.linalg.norm(np.asarray(parent.getImagePos(f, "l")) - ftPoint)
                            points[1] = np.linalg.norm(np.asarray(parent.getImagePos(f, "c")) - ftPoint)
                            points[2] = np.linalg.norm(np.asarray(parent.getImagePos(f, "r")) - ftPoint)
                            hist[np.argmin(points)] += 1

                        if hist.sum() > 0:
                            pointType = indToPoint[np.argmax(hist)]

                    newRow['cam{0}_x'.format(parent.cam)] = parent.getImagePos(f, pointType)[0]
                    newRow['cam{0}_y'.format(parent.cam)] = parent.getImagePos(f, pointType)[1]

                    newRow['cam{0}_tl_x'.format(parent.cam)] = parent.getBoundingBox(f)[0]
                    newRow['cam{0}_tl_y'.format(parent.cam)] = parent.getBoundingBox(f)[1]
                    newRow['cam{0}_c_x'.format(parent.cam)] = parent.getBoundingBox(f)[2]
                    newRow['cam{0}_c_y'.format(parent.cam)] = parent.getBoundingBox(f)[3]
                    newRow['cam{0}_w'.format(parent.cam)] = parent.getBoundingBox(f)[4]
                    newRow['cam{0}_h'.format(parent.cam)] = parent.getBoundingBox(f)[5]
                    newRow['cam{0}_theta'.format(parent.cam)] = parent.getBoundingBox(f)[6]
                    newRow['cam{0}_aa_tl_x'.format(parent.cam)] = parent.getBoundingBox(f)[7]
                    newRow['cam{0}_aa_tl_y'.format(parent.cam)] = parent.getBoundingBox(f)[8]
                    newRow['cam{0}_aa_w'.format(parent.cam)] = parent.getBoundingBox(f)[9]
                    newRow['cam{0}_aa_h'.format(parent.cam)] = parent.getBoundingBox(f)[10]
                    newRow['cam{0}_frame'.format(parent.cam)] = parent.getVideoFrame(f)

                    df = df.append(newRow)
            csv = csv.append(df)

            # Remove used tracklets
            toBeRemoved.append(p)
            cam1 = tm.camIdMap[tm.graph.nodes[p]["cam1"]]
            cam2 = tm.camIdMap[tm.graph.nodes[p]["cam2"]]
            for e in (cam1 + cam2):
                if (e not in toBeRemoved):
                    toBeRemoved.append(e)
        for e in toBeRemoved:
            if (tm.graph.has_node(e)):
                tm.graph.remove_node(e)
        mergedCount += 1
    csv = csv.sort_values(by=['id', 'frame'], ascending=[True, True])
    # Drop cases with exact same frame, id, and x/y coordinates, for each camera view
    csv = csv.drop_duplicates(['frame', 'id', 'cam1_x', 'cam1_y'])
    csv = csv.drop_duplicates(['frame', 'id', 'cam2_x', 'cam2_y'])
    csv.reset_index(inplace=True, drop=True)
    csv, drop_idx = combine2DTracklets(csv, tm)
    csv = csv.drop(drop_idx)
    csv = csv.sort_values(by=['id', 'frame'], ascending=[True, True])
    csv.reset_index(inplace=True, drop=True)
    # Find cases where there are several rows for the same frame in a single Tracklet,
    # and determines which ones minimize the 3D distance (and therefore should be kept)
    csv = csv.drop(getDropIndecies(csv, True))
    return csv


if __name__ == '__main__':

    # seq_names = ['ZebraFish_01', 'ZebraFish_02', 'ZebraFish_03', 'ZebraFish_04']
    # seq_names = ['ZebraFish_08']
    seq_names = ['ZebraFish_05', 'ZebraFish_06', 'ZebraFish_07', 'ZebraFish_08']
    top_view_path = '/data/data0/liuyiran/Zef3d/eval_result/gt_top58/'
    front_view_path = '/data/data0/liuyiran/Zef3d/eval_result/gt_front58/'
    result_root = '/data/data0/liuyiran/Zef3d/eval_result/'
    mathod = 'offset_0'
    track_time_3d = []
    for seq_name in seq_names:
        # A .csv for a video
        cams = ['cam1', 'cam2']
        for cam in cams:
            if cam =='cam1':
                df = txt2dataframe(seq_name, cam, top_view_path)
            else:
                df = txt2dataframe(seq_name, cam, front_view_path)
            csv_filename = result_root + seq_name + "/" + "tracklets_2d_" + cam + '.csv'
            df.to_csv(csv_filename, index=False)
        # start match
        setting_root = result_root + seq_name + "/"
        start = time.time()
        tm = TrackletMatcher(setting_root)
        tracks = tracklet_match(tm)
        track_time_3d.append(time.time()-start)
        # match end
        outputPath = os.path.join(result_root, seq_name, 'tracklets_3d.csv')
        print("Saving data to: {0}".format(outputPath))
        tracks.to_csv(outputPath)
        # got tracklets_3d
        ap = argparse.ArgumentParser(description="Calcualtes the MOT metrics")
        ap.add_argument("-detCSV", "--detCSV", type=str, help="Path to tracking CSV file", default=outputPath)
        gt_path = result_root + "gts/" + seq_name + '.csv'
        ap.add_argument("-gtCSV", "--gtCSV", type=str, help="Path to ground truth CSV file", default=gt_path)

        ap.add_argument("-task", "--task", type=str, default='3D')
        ap.add_argument("-bboxCenter", "--bboxCenter", action="store_true",
                        help="Use the bbox center instead of the head position for cam1 and cam2 tasks")
        ap.add_argument("-useMOTFormat", "--useMOTFormat", action="store_true",
                        help="Use the MOT Challenge ground truth format")
        ap.add_argument("-thresh", "--thresh", type=float, help="Distance threshold", default=10)

        ap.add_argument("-outputFile", "--outputFile", type=str, help="Name of the output file",
                        default="MOT_Metrics.txt")
        result_path = result_root + mathod + "/" + seq_name
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        ap.add_argument("-outputPath", "--outputPath", type=str, help="Path to where the output file should be saved",
                        default=result_path)
        args = vars(ap.parse_args())

        MOT_Evaluation(args)
    print('all time', track_time_3d)