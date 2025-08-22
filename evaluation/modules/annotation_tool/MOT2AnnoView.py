import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def read_gt(txt_path):
    txt_dict = dict()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            img_id = linelist[0]
            obj_id = float(linelist[1])
            bbox = [float(linelist[2]), float(linelist[3]),
                    float(linelist[4]), float(linelist[5]), int(obj_id)]
            if int(img_id) in txt_dict:
                txt_dict[int(img_id)].append(bbox)
            else:
                txt_dict[int(img_id)] = list()
                txt_dict[int(img_id)].append(bbox)
    return txt_dict


def calculate_iou(box1, box2):
    # Bounding box format: (x, y, width, height)

    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2

    # Calculate coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Check if there is no intersection
    if w_intersection <= 0 or h_intersection <= 0:
        return 0.0

    # Calculate areas of bounding boxes and intersection
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    area_intersection = w_intersection * h_intersection

    # Calculate IoU
    iou = area_intersection / (area_box1 + area_box2 - area_intersection)

    return iou


def is_occlude(box, box_list):
    for box1 in box_list:
        iou = calculate_iou(box, box1)
        if iou >= 0.6 and iou != 1:
            return 1
        else:
            continue
    return 0


def xywh_to_xyxy(xywh):
    x, y, w, h, id = xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2, id]


def calculate_center(xywh_box):
    x, y, w, h, _ = xywh_box
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y


if __name__ == '__main__':
    gt_path = '/data/data0/liuyiran/Zef3d/2DTo3D/gts/ZebraFish-08/camF.txt'
    txt_dict = read_gt(gt_path)
    columns = ['Frame', 'Object ID', 'X', 'Y', 'Upper left corner X', 'Upper left corner Y', 'Lower right corner X',
               'Lower right corner Y', 'Occlusion']
    df = pd.DataFrame(columns=columns)
    fish_num = 10
    # same image
    for img_id in sorted(txt_dict.keys()):
        # bbox = x, y, w, h
        for bbox in txt_dict[img_id]:
            occlusion = is_occlude(bbox, txt_dict[img_id])
            center = calculate_center(bbox)
            bbox_xyxy = xywh_to_xyxy(bbox)
            if bbox[4] == 4:
                bbox[4] = 9
            elif bbox[4] == 5:
                bbox[4] = 7
            elif bbox[4] == 7:
                bbox[4] = 8
            elif bbox[4] == 8:
                bbox[4] = 4
            elif bbox[4] == 9:
                bbox[4] = 10
            elif bbox[4] == 10:
                bbox[4] = 5
            row = [img_id, bbox[4], center[0], center[1], bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2],
                   bbox_xyxy[3], occlusion]
            df = df.append(pd.Series(row, index=columns), ignore_index=True)
        if len(txt_dict[img_id]) < fish_num:
            print(len(txt_dict[img_id]), img_id)
    csv_filename = '/data/data0/liuyiran/Zef3d/2DTo3D/bounding_boxes_ZebraFish_08F.csv'
    df.to_csv(csv_filename, index=False)