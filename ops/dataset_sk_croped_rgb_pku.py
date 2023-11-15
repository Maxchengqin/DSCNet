
import time
import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import cv2


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff
        self._parse_list()

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_name):
        # ts = time.time()
        video_name = video_name + '.avi'
        video_path = os.path.join(self.root_path, video_name)
        capture = cv2.VideoCapture(video_path)
        # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        all_images = []
        use_images = []
        # use_idxs = []
        frame_count = 0
        success = True
        while success:
            success, frame = capture.read()
            if frame is None:
                break
            frame_count += 1
            # frame = cv2.resize(frame, (512, 424))
            # image = Image.fromarray(frame)
            all_images.append(frame)

        average_duration = (frame_count - self.new_length) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        # elif frame_count > self.num_segments and (frame_count + 1) > self.new_length:
        elif frame_count > self.num_segments and frame_count > self.new_length:
            # offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            offsets = np.sort(randint(frame_count - self.new_length, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        for stidx in offsets:
            for i in range(self.new_length):
                while stidx + i >= frame_count:
                    i -= 1
                try:
                    # use_images.append(Image.fromarray(all_images[int(stidx + i)]))
                    use_images.append(Image.fromarray(cv2.resize(all_images[int(stidx + i)], (256, 256))))
                except:
                    print('本视频帧数：', frame_count, video_path)
        # te = time.time()
        # print('耗时：', te-ts)
        return use_images

    def _get_val_indices(self, video_name):
        video_name = video_name + '.avi'
        video_path = os.path.join(self.root_path, video_name)
        capture = cv2.VideoCapture(video_path)
        # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        all_images = []
        use_images = []
        # use_idxs = []
        frame_count = 0
        success = True
        while success:
            success, frame = capture.read()
            if frame is None:
                break
            frame_count += 1
            # frame = cv2.resize(frame, (512, 424))
            # image = Image.fromarray(frame)
            all_images.append(frame)
        # if frame_count > self.num_segments and (frame_count + 1) > self.new_length:
        if frame_count > self.num_segments and frame_count > self.new_length:
            # offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            # offsets = np.sort(randint(frame_count - self.new_length, size=self.num_segments))
            tick = (frame_count - self.new_length) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        for stidx in offsets:
            for i in range(self.new_length):
                # use_idxs.append(stidx + i)
                while stidx + i >= frame_count:
                    i -= 1
                try:
                    # use_images.append(Image.fromarray(all_images[int(stidx + i)]))
                    use_images.append(Image.fromarray(cv2.resize(all_images[int(stidx + i)], (256, 256))))
                except:
                    print('本视频帧数：', frame_count)
        return use_images

    def _get_test_indices(self, video_name):
        video_name = video_name + '.avi'
        video_path = os.path.join(self.root_path, video_name)
        capture = cv2.VideoCapture(video_path)
        # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        all_images = []
        use_images = []
        # use_idxs = []
        frame_count = 0
        success = True
        while success:
            success, frame = capture.read()
            if frame is None:
                break
            frame_count += 1
            # frame = cv2.resize(frame, (512, 424))
            # image = Image.fromarray(frame)
            all_images.append(frame)
        # tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        tick = (frame_count - self.new_length) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        for stidx in offsets:
            for i in range(self.new_length):
                # use_idxs.append(stidx + i)
                while stidx + i >= frame_count:
                    i -= 1
                try:
                    # use_images.append(Image.fromarray(all_images[int(stidx + i)]))
                    use_images.append(Image.fromarray(cv2.resize(all_images[int(stidx + i)], (256, 256))))
                except:
                    print('本视频帧数：', frame_count)
        return use_images

    def __getitem__(self, index):
        record = self.video_list[index]
        label = record.label
        # check this is a legit video folder

        if not self.test_mode:
            use_images = self._sample_indices(record.path) if self.random_shift else self._get_val_indices(record.path)
        else:
            use_images = self._get_test_indices(record.path)
        process_data = self.transform(use_images)
        return process_data, label

    def __len__(self):
        return len(self.video_list)
