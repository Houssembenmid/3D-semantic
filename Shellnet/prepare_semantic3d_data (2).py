#!/usr/bin/python3
'''Prepare Data for Semantic3D Segmentation Task.'''
# we can change either block size or grid size within each block 
# block splits 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

def read_points(f):
    # reads Semantic3D .txt file f into a pandas dataframe
    col_names = ['x', 'y', 'z', 'i', 'r', 'g', 'b']
    col_dtype = {'x': np.float32, 'y': np.float32, 'z': np.float32, 'i': np.int32,
                  'r': np.uint8, 'g': np.uint8, 'b': np.uint8}
    xyzirgb = pd.read_csv(f, names=col_names, dtype=col_dtype, delim_whitespace=True)
    xyzirgb = xyzirgb.to_numpy()
    return xyzirgb[:,0:3], xyzirgb[:,4:7]  # xyz, rgb

def load_labels(label_path):
    # Assuming each line is a valid int
    with open(label_path, "r") as f:
        labels = [int(line) for line in f]
    return np.array(labels, dtype=np.int32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--max_point_num', '-m', help='Max point number of each sample', type=int, default=8192)
    parser.add_argument('--block_size', '-b', help='Block size', type=float, default=10)
    parser.add_argument('--grid_size', '-g', help='Grid size', type=float, default=0.2)
    
    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../dataset/semantic_raw/'
    max_point_num = args.max_point_num

    batch_size = 2048
    data = np.zeros((batch_size, max_point_num, 6))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_point_num), dtype=np.int32)

    folders = [os.path.join(root, folder) for folder in ['train', 'val', 'test_reduced']]
    for folder in folders:
        datasets = [filename[:-4] for filename in os.listdir(folder) if filename.endswith('.txt')]
        for dataset_idx, dataset in enumerate(datasets): #return column index and column label
            h5_file_name = os.path.join(folder,dataset + '_half_0.h5')
            if os.path.isfile(h5_file_name):
                continue
            
            filename_txt = os.path.join(folder, dataset + '.txt') #create fullpath of the dataset.txt
            # print('{}-Loading {}...'.format(datetime.now(), filename_txt))
            # xyzirgb = np.loadtxt(filename_txt)
            # print('{}-Loading finished{}...'.format(datetime.now(), filename_txt))

            print('{}-Loading {}...'.format(datetime.now(), filename_txt))
            [xyz, rgb] = read_points(filename_txt) # create 2 arrays eacg is a matrix with shape(num_points,3)
            print('{}-Loading finished{}...'.format(datetime.now(), filename_txt))

            filename_labels = os.path.join(folder, dataset + '.labels')
            has_labels = os.path.exists(filename_labels)
            if has_labels:
                print('{}-Loading {}...'.format(datetime.now(), filename_labels))
                labels = load_labels(filename_labels) # obtain array of labels (point_nm,datatype=int32)
                indices = (labels != 0) # array booloean : array([ True,  True,  True, ...,  True,  True,  True])
                labels = labels[indices] - 1  # since labels == 0 have been removed we start with naturalground with label 0
                xyz = xyz[indices, :]  # remove 0 dataset 
                rgb = rgb[indices, :]  # remove 0 dataset
            else:
                labels = np.zeros((xyz.shape[0])) #set test data labels to zeros

            offsets = [('zero', 0.0), ('half', args.block_size / 2)] #Offset is the position in the dataset of a particular record
            for offset_name, offset in offsets:
                idx_h5 = 0
                idx = 0

                print('{}-Computing block id of {} points...'.format(datetime.now(), xyz.shape[0]))#  xyz.shape[0] =numb of points
                xyz_min = np.amin(xyz, axis=0, keepdims=True) - offset  # min x y and z
                xyz_max = np.amax(xyz, axis=0, keepdims=True)   # max x y and z
                block_size = (args.block_size, args.block_size, 2 * (xyz_max[0, -1] - xyz_min[0, -1]))#block_size : 5x5xmaximum_height
                xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int) #create points that form square blocks of 5m

                print('{}-Collecting points belong to each block...'.format(datetime.now(), xyz.shape[0]))
                blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True,
                                                                            return_counts=True, axis=0)
                # blocks: coordinates that forms a block (x,y,0) 
                #point_block_indices : indices of each point belongs to a block
                # block_point_counts : number of points in each block
                block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
                # this gives the point indices within each block
                print('{}-{} is split into {} blocks.'.format(datetime.now(), dataset, blocks.shape[0]))

                block_to_block_idx_map = dict()
                for block_idx in range(blocks.shape[0]):
                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    block_to_block_idx_map[(block[0], block[1])] = block_idx # create a dict that index each x,y block from 0 to num of blocks

                # merge small blocks into one of their big neighbors
                block_point_count_threshold = max_point_num / 10
                nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
                block_merge_count = 0
                for block_idx in range(blocks.shape[0]):
                    if block_point_counts[block_idx] >= block_point_count_threshold:
                        continue

                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    for x, y in nbr_block_offsets:
                        nbr_block = (block[0] + x, block[1] + y)
                        if nbr_block not in block_to_block_idx_map:
                            continue

                        nbr_block_idx = block_to_block_idx_map[nbr_block]
                        if block_point_counts[nbr_block_idx] < block_point_count_threshold:
                            continue

                        block_point_indices[nbr_block_idx] = np.concatenate(
                            [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
                        block_point_indices[block_idx] = np.array([], dtype=np.int)
                        block_merge_count = block_merge_count + 1
                        break
                print('{}-{} of {} blocks are merged.'.format(datetime.now(), block_merge_count, blocks.shape[0]))

                idx_last_non_empty_block = 0
                for block_idx in reversed(range(blocks.shape[0])):
                    if block_point_indices[block_idx].shape[0] != 0:
                        idx_last_non_empty_block = block_idx
                        break      # check all blocks are not empty

                # uniformly sample each block
                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = block_point_indices[block_idx]  #point_indices :point_indices in a single block
                    if point_indices.shape[0] == 0:  #if a block is empty
                        continue
                    block_points = xyz[point_indices] # coordonates of block indices
                    block_min = np.amin(block_points, axis=0, keepdims=True)
                    xyz_grids = np.floor((block_points - block_min) / args.grid_size).astype(np.int)
                    grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                             return_counts=True, axis=0)
                    # divide each block into grid same way before
                    grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
                    # this gives the point indices within each block 
                    grid_point_count_avg = int(np.average(grid_point_counts)) #average ptnumber within grids in a block
                    point_indices_repeated = []
                    for grid_idx in range(grids.shape[0]):
                        point_indices_in_block = grid_point_indices[grid_idx] #point_indices_in_grid
                        repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0]) 
                        #repeat_num returns points multiple of the average points within each grid (c'est l'inverse!!!)
                        if repeat_num > 1: # if the num of points within each grid is below the average
                            point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)  #data augmentation
                            np.random.shuffle(point_indices_in_block) #shuffle point indices in a grid that has repeated points
                            point_indices_in_block = point_indices_in_block[:grid_point_count_avg] 
                            # take the average point number of grids within each block after repetition and shuffling
                        point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
                    block_point_indices[block_idx] = np.array(point_indices_repeated) # get original indexes of the points after repeating and shuffling
                    block_point_counts[block_idx] = len(point_indices_repeated) # new num of points within each block

                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = block_point_indices[block_idx]
                    if point_indices.shape[0] == 0:
                        continue

                    block_point_num = point_indices.shape[0]  # num points within each block
                    block_split_num = int(math.ceil(block_point_num * 1.0 / max_point_num))# multiplication of max point num (rounded)
                    point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num)) # average point number 
                    point_nums = [point_num_avg] * block_split_num
                    point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
                    starts = [0] + list(np.cumsum(point_nums))

                    np.random.shuffle(point_indices)
                    block_points = xyz[point_indices]
                    block_min = np.amin(block_points, axis=0, keepdims=True)
                    block_max = np.amax(block_points, axis=0, keepdims=True)
                    block_center = (block_min + block_max) / 2
                    block_center[0][-1] = block_min[0][-1]
                    block_points = block_points - block_center  # align to block bottom center (centrelize each block)
                    x, y, z = np.split(block_points, (1, 2), axis=-1)
                    block_xzyrgbi = np.concatenate([x, z, y, rgb[point_indices]], axis=-1) # matrix in that order in e given block
                    block_labels = labels[point_indices] # label of each point in a bgiven lock

                    for block_split_idx in range(block_split_num):
                        start = starts[block_split_idx]
                        point_num = point_nums[block_split_idx] # point number in ach set of points in a block after spliting the total number of points in a block on the max point number which is 8192 and round it
                        end = start + point_num # last point in a given split
                        idx_in_batch = idx % batch_size
                        data[idx_in_batch, 0:point_num, ...] = block_xzyrgbi[start:end, :]# fill data with point values blocksplitwise
                        data_num[idx_in_batch] = point_num
                        label[idx_in_batch] = dataset_idx  # won't be used...
                        label_seg[idx_in_batch, 0:point_num] = block_labels[start:end]#labels of each point in a split of a block
                        indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end] # indices of each of that label
                        

                        if ((idx + 1) % batch_size == 0) or \
                                (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1):
                            item_num = idx_in_batch + 1
                            filename_h5 = os.path.join(folder, dataset + '_%s_%d.h5' % (offset_name, idx_h5))
                            print('{}-Saving {}...'.format(datetime.now(), filename_h5))

                            file = h5py.File(filename_h5, 'w')
                            file.create_dataset('data', data=data[0:item_num, ...]) # data contins x ,z ,y centred and rgb
                            file.create_dataset('data_num', data=data_num[0:item_num, ...]) #num of points within each block
                            file.create_dataset('label', data=label[0:item_num, ...]) # won't be used
                            file.create_dataset('label_seg', data=label_seg[0:item_num, ...]) # label of corresponding data
                            file.create_dataset('indices_split_to_full', data=indices_split_to_full[0:item_num, ...]) #original indices of the data
                            file.close()

                            idx_h5 = idx_h5 + 1
                        idx = idx + 1


if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))