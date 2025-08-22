import os
import numpy as np

# convert the gts of 3dzef to mot

if __name__ == '__main__':

  DATA_PATH = '/data/data0/liuyiran/dataset/3dzef/test/'
  OUT_PATH = DATA_PATH + 'new_gts/'
  SPLITS = ['train']

  for split in SPLITS:
    data_path = DATA_PATH
    seqs = ['ZebraFish-01', 'ZebraFish-02', 'ZebraFish-03', 'ZebraFish-04']
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
      if '.DS_Store' in seq:
        continue
      video_cnt += 1
      seq_path = '{}/{}/'.format(data_path, seq)
      img_path = seq_path + 'imgF/'
      ann_path = seq_path + 'gt/gt.txt'
      output_path = seq_path + 'gt/gt_F.txt'
      # images = os.listdir(img_path)
      # num_images = len([image for image in images if 'jpg' in image])
      anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
      idx = np.lexsort(anns.T[:2, :])
      gt = anns[idx, :]
      # for fid, tid, _, _, _, _, _, camT_left, camT_top, camT_width, camT_height, _, _, _, _, _, _, _, _ in gt:
      for fid, tid, _, _, _, _, _, _, _, _, _, _, _, _, camF_left, camF_top, camF_width, camF_height, _ in gt:
        labels ='{:d},{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:d},{:d},{:d}\n'\
          .format(int(fid), int(tid), camF_left, camF_top, camF_width, camF_height, 1, 1, 1)
        with open(output_path, 'a') as f:
            f.write(labels)
