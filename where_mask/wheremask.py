import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import h5py
import cv2
from PIL import Image
import numpy as np
import re
import collections
import argparse

parser = argparse.ArgumentParser()

def create_grid(x, y, startx, endx, starty, endy):  
  grid = [[0]*x for _ in range(y)]
  for i in range(startx, endx):
    for j in range(starty , endy):
      grid[i][j] = 1
  return grid

def get_num_keys(destination_path):
  fcheck = h5py.File(destination_path, 'r')
  print(len(fcheck.keys()))
  return len(fcheck.keys())


def get_dict(dicts, coords, objects):
  """
  return the coordinate of the newly added object
  """
  o_idx = [i for i, x in enumerate(objects) if x == 1]
  # print(o_idx)
  # print(dicts)
  for idx in o_idx:
    if idx not in dicts:
      dicts[idx] = coords[idx]
      return coords[idx]

def get_quadrant(coords):
  x, y, _ = coords
  if x == 64 and y == 64:
    return 0
  elif x >= 64 and y >= 64:
    return 4
  elif x >= 64 and y <= 64:
    return 1
  elif x <= 64 and y <= 64:
    return 2
  elif x <= 64 and y >= 64:
    return 3 


parser.add_argument("--original_path", default= '/GeNeVA_datasets/data/iCLEVR/clevr_train.h5')
parser.add_argument("--destination_path", default= 'clevr_train_where_mask.h5')


if __name__ == "__main__":

  args = parser.parse_args()

  shapex = 128
  shapey = 128
  grid_dicts = {}
  grid0 = create_grid(shapex, shapey, 0,0,0,0)
  grid_dicts[0] = grid0 
  grid1 = create_grid(shapex, shapey, 0, shapex//2, shapex//2, shapex)
  grid_dicts[1] = grid1 
  grid2 = create_grid(shapex, shapey, 0, shapex//2, 0, shapex//2)
  grid_dicts[2] = grid2
  grid3 = create_grid(shapex, shapey, shapex//2, shapex, 0, shapex//2)
  grid_dicts[3] = grid3
  grid4 = create_grid(shapex, shapey, shapex//2, shapex, shapex//2, shapex)
  grid_dicts[4] = grid4

  original_path = args.original_path
  destination_path = args.destination_path

  #create a new skeleton to add masks 
  fs = h5py.File(original_path, 'r')
  fd = h5py.File(destination_path, 'w')

  #get number of datasets in h5py file
  batch_size_copy = get_num_keys(original_path) -2

  for i in range(batch_size_copy):
    xid = f'{i:06d}'
    fs.copy(xid,fd)

  fs.copy('background', fd)
  fs.copy('entities', fd)
  fs.close()
  fd.close()

  #populate the new masks to the generated h5 file
  destination_path = destination_path
  fdest = h5py.File(destination_path, 'r+')
  num_keys_in_destination = get_num_keys(destination_path) - 2
  count = 0
  for i in (range(num_keys_in_destination)):
    count += 1
    print(count ,'/', num_keys_in_destination)
    xid = f'{i:06d}'
    # img_path = fdest[xid]['images']
    dicts = {}
    where_mask = []
    for i in range(5):
      coord_of_object = get_dict(dicts, fdest[xid]['coords'].value[i], fdest[xid]['objects'].value[i])
      # print(coord_of_object)
      quad = get_quadrant(coord_of_object)
      # print(quad, 'QUAD')
      if i > 0:
        mask_grid = grid_dicts[quad]
      else:
        mask_grid = grid_dicts[0]
      where_mask.append(mask_grid)
      # plt.imshow(mask_grid)
      # plt.show()
      # cv2_imshow(img_path.value[i])
    fdest[xid].create_dataset('where', data = np.array(where_mask))








