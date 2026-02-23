import os
import numpy as np
import torch
import concurrent.futures
import time


# inspired from:
#   https://jamesmccaffrey.wordpress.com/2021/03/08/working-with-huge-training-data-files-for-pytorch/


class ReplayDfStreamerxyz:
    def __init__(self, df_dir, bat_size, gamemode, shuffle=False, bat_density=1, remove_neutral_labels=False, z_threshold=0):  # make sure df_dir has a slash at the end
        self.df_dir = df_dir
        self.bat_size = bat_size
        self.bat_density = 1/bat_density  # inverted here to use floor division later on
        self.remove_neutral_labels = remove_neutral_labels
        self.z_threshold = z_threshold
        self.shuffle = shuffle
        self.gamemode = gamemode
        if self.gamemode == "duel":
            self.arraywidth = 115
        elif self.gamemode == "doubles":
            self.arraywidth = None
        elif self.gamemode == "standard":
            self.arraywidth = None
        else:
            self.arraywidth = None

        self.batch_counter = 0

        self.filenames = os.listdir(df_dir)
        self.num_replays = len(self.filenames)
        self.filepaths = []
        for file in self.filenames:
            der = df_dir + file
            self.filepaths.append(der)
        if self.shuffle:
            np.random.shuffle(self.filepaths)  # randomize order of filepaths

        # chunk up randomized filepaths
        # due to the way np.array_split works, the batch size doesn't need to divide evenly into the num_files
        # however, this will result in slightly uneven batch sizes
        # e.g. 1007 files with a batch size of 100 will result in 10 batches, 7 of them being size 101, 3 being size 100
        if self.bat_size >= self.num_replays:
            self.batchedfiles = [np.array(self.filepaths)]
        else:
            self.batchedfiles = np.array_split(self.filepaths, (self.num_replays / self.bat_size))
        self.num_batches = len(self.batchedfiles)
        self.batches_remaining = self.num_batches
        self.current_files = self.batchedfiles[self.batch_counter]

        self.x_data = None
        self.y_data = None
        self.z_data = None


    def load_batch(self):

        self.x_data = None
        self.y_data = None
        self.z_data = None
        torch.cuda.empty_cache()


        if self.batch_counter >= self.num_batches:
            return False  # done with batches

        batch = self.batchedfiles[self.batch_counter]
        self.current_files = batch

        xyz = []

        if self.bat_density != 1:
            shufflearray = np.arange(100000) # for preshuffling
            np.random.shuffle(shufflearray)
            for i, file in enumerate(batch):
                array = np.load(file)
                fileshuffler = shufflearray[shufflearray < len(array)]
                array = array[fileshuffler]
                densitycutindex = (int(len(array)//self.bat_density))
                array = array[0:densitycutindex, :]
                array = torch.from_numpy(array).to('cuda')
                xyz.append(array)
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for array in executor.map(np.load, batch):
                    array = torch.from_numpy(array).to('cuda')
                    xyz.append(array)
        xyz = torch.vstack(xyz)
        if self.shuffle:
            xyz = xyz[torch.randperm(xyz.size()[0])]  # 10-20x faster to shuffle this on the gpu

        if self.z_threshold > 0:
            z = xyz[:, -1]
            xyz = xyz[torch.where(z <= self.z_threshold)]

        x = xyz[:, :-2]
        y = xyz[:, -2]
        z = xyz[:, -1]
        del xyz

        if self.remove_neutral_labels:
            binary_idxs = torch.where(y != 0.5)
            x = x[binary_idxs]
            y = y[binary_idxs]
            z = z[binary_idxs]
            #print(f"removed {len(xyz)-len(y)} neutral labels.")


        self.x_data, self.y_data, self.z_data = x, y, z
        self.batch_counter += 1
        return True

    def __iter__(self):
        return self

    def __next__(self):
        reload = self.load_batch()
        if not reload:
            return False
        x = self.x_data
        y = self.y_data
        z = self.z_data
        self.batches_remaining = self.num_batches - self.batch_counter
        return x, y, z

