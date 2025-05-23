import random
import torch.utils.data
from pytracking import TensorDict
import copy

def get_united_bbox(boxes, start_idx, end_idx):
    # xywh to xyxy
    new_boxes = copy.deepcopy(boxes[start_idx:end_idx + 1])
    new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]
    new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]
    xmin, ymin, xmax, ymax = new_boxes[:, 0].min().item(), new_boxes[:, 1].min().item(), new_boxes[:,
                                                                                         2].max().item(), new_boxes[
                                                                                                          :,
                                                                                                          3].max().item()
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def get_aligned_bbox(boxes):
    aligned_bbox = []
    for i in range(len(boxes) - 1):
        aligned_bbox.append(torch.tensor(get_united_bbox(boxes, i, i + 1)))
    aligned_bbox = torch.stack(aligned_bbox, dim=0).to(boxes.device)
    return aligned_bbox

def get_raw_bbox(boxes):
    return boxes[:-1]

def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of test frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
            elif self.frame_sample_mode == 'continual':
                # Sample test frames and train frames in a continual manner, i.e. test_frame_ids and train_frame_ids are continual
                base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                idx_valid = False
                while not idx_valid:
                    train_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                             min_id=base_frame_id[
                                                        0] - self.max_gap - gap_increase,
                                             max_id=base_frame_id[
                                                        0] + self.max_gap + gap_increase - self.num_train_frames + 1)
                    train_frame_ids = [i for i in range(train_frame_start_idx[0],
                                                         train_frame_start_idx[0] + self.num_train_frames)]
                    if train_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[train_frame_ids].all()
                idx_valid = False
                while not idx_valid:
                    test_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                                                min_id=base_frame_id[
                                                                        0] - self.max_gap - gap_increase,
                                                                max_id=base_frame_id[
                                                                           0] + self.max_gap + gap_increase - self.num_test_frames + 1)
                    test_frame_ids = [i for i in range(test_frame_start_idx[0], test_frame_start_idx[0] + self.num_test_frames)]
                    if test_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[test_frame_ids].all()            
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        # print(f'train_ids:{train_frame_ids}, test_ids:{test_frame_ids}')
        train_frames, train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        test_frames, test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        # data = TensorDict({'train_images': train_frames,
        #                    'train_anno': train_anno['bbox'],
        #                    'test_images': test_frames,
        #                    'test_anno': test_anno['bbox'],
        #                    'dataset': dataset.get_name(),
        #                    'test_class': meta_obj_test.get('object_class_name')})
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name(),
                           })

        # return self.processing(data)
        if self.processing is not None:
            data['train_images'] = list(map(lambda x: x.transpose(1, 2, 0), data['train_images']))
            data['test_images'] = list(map(lambda x: x.transpose(1, 2, 0), data['test_images']))
            data = self.processing(data)
        return data


class TransTBaseTrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of a set of template frames and search frames, used to train the TransT model.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'template frames' and
    'search frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval', quantization_layer=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the search frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.quantization_layer = quantization_layer

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible) - 1:   ### abandon the last frame
            max_id = len(visible) - 1

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)


    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False

        # Sample a sequence
        seq_id = random.randint(0, dataset.get_num_sequences() - 1)

        # Sample frames
        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']


        if is_video_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_template_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                              min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                             max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
            elif self.frame_sample_mode == 'continual':
                # Sample search frames and template frames in a continual manner, i.e. search_frame_ids and template_frame_ids are continual
                base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                idx_valid = False
                while not idx_valid:
                    template_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                             min_id=base_frame_id[
                                                        0] - self.max_gap - gap_increase,
                                             max_id=base_frame_id[
                                                        0] + self.max_gap + gap_increase - self.num_template_frames + 1)
                    template_frame_ids = [i for i in range(template_frame_start_idx[0],
                                                         template_frame_start_idx[0] + self.num_template_frames)]
                    if template_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[template_frame_ids].all()
                idx_valid = False
                while not idx_valid:
                    search_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                                                min_id=base_frame_id[
                                                                        0] - self.max_gap - gap_increase,
                                                                max_id=base_frame_id[
                                                                           0] + self.max_gap + gap_increase - self.num_search_frames + 1)
                    search_frame_ids = [i for i in range(search_frame_start_idx[0], search_frame_start_idx[0] + self.num_search_frames)]
                    if search_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[search_frame_ids].all()
                # print(f'base_frame_id: {base_frame_id}, search_frame_ids: {search_frame_ids}')
                # Increase gap until a frame is found
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames

        if self.num_search_frames == 1:
            is_extend = True
        else:
            is_extend = False
        # first get events to avoiding the extended frame ids
        template_events = dataset.raw_event_loader(seq_id, template_frame_ids)
        search_events = dataset.raw_event_loader(seq_id, search_frame_ids)
        template_frames, template_anno, meta_obj_template = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict, is_extend)
        search_frames, search_anno, meta_obj_search = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict, is_extend)
        if self.quantization_layer is not None:
            template_frames = self.quantization_layer(torch.from_numpy(template_events)).numpy()
            search_frames = self.quantization_layer(torch.from_numpy(search_events)).numpy()
        # print(f'template frame shape: {template_frames.shape}, search frame shape: {search_frames.shape}')
        # if self.num_search_frames == 1:
        #     template_frames = template_frames[0]
        #     template_anno['bbox'] = template_anno['bbox'][0]
        #     search_frames = search_frames[0]
        #     search_anno['bbox'] = search_anno['bbox'][0]
        data = TensorDict({'template_images': template_frames,
                           'template_anno': template_anno['bbox'],
                           'search_images': search_frames,
                           'search_anno': search_anno['bbox'],
                           'template_events': template_events,
                           'search_events': search_events})
        if self.num_search_frames == 1:
            ## align
            data['template_anno'] = get_aligned_bbox(data['template_anno'])
            data['search_anno'] = get_aligned_bbox(data['search_anno'])
            ## raw
            # data['template_anno'] = get_raw_bbox(data['template_anno'])
            # data['search_anno'] = get_raw_bbox(data['search_anno'])
        if self.processing is not None:
            data['template_images'] = list(map(lambda x: x.transpose(1, 2, 0), data['template_images']))
            data['search_images'] = list(map(lambda x: x.transpose(1, 2, 0), data['search_images']))
            data = self.processing(data)
        if self.num_search_frames == 1:
            data['template_images'] = data['template_images'][0]
            data['template_anno'] = data['template_anno'][0]
            data['search_images'] = data['search_images'][0]
            data['search_anno'] = data['search_anno'][0]
        return data


class TransTSampler(TransTBaseTrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval', quantization_layer=None):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_search_frames=num_search_frames, num_template_frames=num_template_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode, quantization_layer=quantization_layer)
        

class DiMPSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


class ATOMSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames=1, num_train_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


class LWLSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames and ii) a set of test frames. The train frames, along with the
    ground-truth masks, are passed to the few-shot learner to obtain the target model parameters \tau. The test frames
    are used to compute the prediction accuracy.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is randomly
    selected from that dataset. A base frame is then sampled randomly from the sequence. The 'train frames'
    are then sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id], and the 'test frames'
    are sampled from the sequence from the range (base_frame_id, base_frame_id + max_gap] respectively. Only the frames
    in which the target is visible are sampled. If enough visible frames are not found, the 'max_gap' is increased
    gradually until enough frames are found. Both the 'train frames' and the 'test frames' are sorted to preserve the
    temporal order.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, p_reverse=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            p_reverse - Probability that a sequence is temporally reversed
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing

        self.p_reverse = p_reverse

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        reverse_sequence = False
        if self.p_reverse is not None:
            reverse_sequence = random.random() < self.p_reverse

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (self.num_test_frames + self.num_train_frames)

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
            while test_frame_ids is None:
                if gap_increase > 1000:
                    raise Exception('Frame not found')

                if not reverse_sequence:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
                else:
                    # Sample in reverse order, i.e. train frames come after the test frames
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_test_frames + 1,
                                                             max_id=len(visible) - self.num_train_frames - 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0],
                                                              max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=0,
                                                              max_id=train_frame_ids[0] - 1,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        # Sort frames
        train_frame_ids = sorted(train_frame_ids, reverse=reverse_sequence)
        test_frame_ids = sorted(test_frame_ids, reverse=reverse_sequence)

        all_frame_ids = train_frame_ids + test_frame_ids

        # Load frames
        all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)

        train_frames = all_frames[:len(train_frame_ids)]
        test_frames = all_frames[len(train_frame_ids):]

        train_anno = {}
        test_anno = {}
        for key, value in all_anno.items():
            train_anno[key] = value[:len(train_frame_ids)]
            test_anno[key] = value[len(train_frame_ids):]

        train_masks = train_anno['mask'] if 'mask' in train_anno else None
        test_masks = test_anno['mask'] if 'mask' in test_anno else None

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name()})

        return self.processing(data)


class KYSSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            sequence_sample_info - A dict containing information about how to sample a sequence, e.g. number of frames,
                                    max gap between frames, etc.
            processing - An instance of Processing class which performs the necessary processing of the data.
            sample_occluded_sequences - If true, sub-sequence containing occlusion is sampled whenever possible
        """

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, valid, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(valid):
            max_id = len(valid)

        valid_ids = [i for i in range(min_id, max_id) if valid[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def find_occlusion_end_frame(self, first_occ_frame, target_not_fully_visible):
        for i in range(first_occ_frame, len(target_not_fully_visible)):
            if not target_not_fully_visible[i]:
                return i

        return len(target_not_fully_visible)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        p_datasets = self.p_datasets

        dataset = random.choices(self.datasets, p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_test_frames = self.sequence_sample_info['num_test_frames']
        max_train_gap = self.sequence_sample_info['max_train_gap']
        allow_missing_target = self.sequence_sample_info['allow_missing_target']
        min_fraction_valid_frames = self.sequence_sample_info.get('min_fraction_valid_frames', 0.0)

        if allow_missing_target:
            min_visible_frames = 0
        else:
            raise NotImplementedError

        valid_sequence = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not valid_sequence:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            visible_ratio = seq_info_dict.get('visible_ratio', visible)

            num_visible = visible.type(torch.int64).sum().item()

            enough_visible_frames = not is_video_dataset or (num_visible > min_visible_frames and len(visible) >= 20)

            valid_sequence = enough_visible_frames

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                gap_increase = 0

                test_valid_image = torch.zeros(num_test_frames, dtype=torch.int8)
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    occlusion_sampling = False
                    # if dataset.has_occlusion_info() and self.sample_occluded_sequences:
                    #     target_not_fully_visible = visible_ratio < 0.9
                    #     if target_not_fully_visible.float().sum() > 0:
                    #         occlusion_sampling = True

                    if occlusion_sampling:
                        first_occ_frame = target_not_fully_visible.nonzero()[0]

                        occ_end_frame = self.find_occlusion_end_frame(first_occ_frame, target_not_fully_visible)

                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=max(0, first_occ_frame - 20),
                                                         max_id=first_occ_frame - 5)

                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)

                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        end_frame = min(occ_end_frame + random.randint(5, 20), len(visible) - 1)

                        if (end_frame - base_frame_id) < num_test_frames:
                            rem_frames = num_test_frames - (end_frame - base_frame_id)
                            end_frame = random.randint(end_frame, min(len(visible) - 1, end_frame + rem_frames))
                            base_frame_id = max(0, end_frame - num_test_frames + 1)

                            end_frame = min(end_frame, len(visible) - 1)

                        step_len = float(end_frame - base_frame_id) / float(num_test_frames)

                        test_frame_ids = [base_frame_id + int(x * step_len) for x in range(0, num_test_frames)]
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0] * (num_test_frames - len(test_frame_ids))
                    else:
                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=2*num_train_frames,
                                                         max_id=len(visible) - int(num_test_frames * min_fraction_valid_frames))
                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)
                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        test_frame_ids = list(range(base_frame_id, min(len(visible), base_frame_id + num_test_frames)))
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0]*(num_test_frames - len(test_frame_ids))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Get frames
        # print(f'train_ids:{train_frame_ids}, test_ids:{test_frame_ids}')
        train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_valid_image': test_valid_image,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})

        # Send for processing
        # return self.processing(data)
        if self.processing is not None:
            data['train_images'] = list(map(lambda x: x.transpose(1, 2, 0), data['train_images']))
            data['test_images'] = list(map(lambda x: x.transpose(1, 2, 0), data['test_images']))
            data = self.processing(data)
        return data

class SequentialTargetCandidateMatchingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, samples_per_epoch, sup_modes, p_sup_modes=None, processing=no_processing,
                 subseq_modes=None, p_subseq_modes=None, frame_modes=None, p_frame_modes=None):
        """
        args:
            datasets - List of datasets to be used for training
            samples_per_epoch - Number of training samples per epoch
            sup_modes - List of different supervision modes to use (partial_sup or self_sup).
            p_sup_modes - List of sup_mode sample probabilities.
            processing - An instance of Processing class which performs the necessary processing of the data.
            subseq_modes - List of different subsequence modes to sample from (HH, HK, HG), see KeepTrack paper for details.
            p_subseq_modes - List of subseq_mode sample probabilities.
            frame_modes - List of different frame mode to sample from (H, K, J), see KeepTrack paper for details.
            p_frame_modes - List of frame_mode sample probabilities.

        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.processing = processing
        self.subseq_modes = subseq_modes
        self.frame_modes = frame_modes
        self.sup_modes = sup_modes if sup_modes is not None else ['self_sup']
        self.p_sup_modes = p_sup_modes

        if p_sup_modes is None:
            self.p_sup_modes = [1. / len(self.sup_modes)] * (len(self.sup_modes))

        if subseq_modes is not None:
            self.dataset_subseq_states = self._load_dataset_subsequence_states()

            if p_subseq_modes is None:
                p_subseq_modes = [self.dataset_subseq_states[mode].shape[0] for mode in self.subseq_modes]

            # Normalize
            p_subseq_total = sum(p_subseq_modes)
            self.p_subseq_modes = [x / p_subseq_total for x in p_subseq_modes]

        if frame_modes is not None:
            self.dataset_frame_states = self._load_dataset_frame_states()

            if p_frame_modes is None:
                p_frame_modes = [self.dataset_frame_states[mode].shape[0] for mode in self.frame_modes]

            # Normalize
            p_frames_total = sum(p_frame_modes)
            self.p_frame_modes = [x / p_frames_total for x in p_frame_modes]

    def __len__(self):
        return self.samples_per_epoch

    def _load_dataset_subsequence_states(self):
        return self.dataset.get_subseq_states()

    def _load_dataset_frame_states(self):
        return self.dataset.get_frame_states()

    def _sample_valid_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which dumped data is useful

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be sampled
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 2:
            min_id = 2
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        num_begin = num_ids//2
        num_end = num_ids - num_ids//2
        ids_begin = random.sample(valid_ids[:len(valid_ids)//2], k=num_begin)
        ids_end = random.sample(valid_ids[len(valid_ids)//2:], k=num_end)
        return ids_begin + ids_end


    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly).

        returns:
            TensorDict - dict containing all the data blocks
        """

        # select a subseq mode
        sup_mode = random.choices(self.sup_modes, self.p_sup_modes, k=1)[0]

        if sup_mode == 'self_sup':
            mode = random.choices(self.frame_modes, self.p_frame_modes, k=1)[0]

            states = self.dataset_frame_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id]

        elif sup_mode == 'partial_sup':
            mode = random.choices(self.subseq_modes, self.p_subseq_modes, k=1)[0]

            states = self.dataset_subseq_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id, baseframe_id + 1]
        else:
            raise ValueError('Supervision mode: \'{}\' is invalid.'.format(sup_mode))


        seq_info_dict = self.dataset.get_sequence_info(seq_id)

        frames_dict, _ = self.dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({
            'dataset': self.dataset.get_name(),
            'mode': mode,
            'seq_name': self.dataset.get_sequence_name(seq_id),
            'base_frame_id': baseframe_id,
            'sup_mode': sup_mode
        })

        for key, val in frames_dict.items():
            data[key] = val

        return self.processing(data)


class TaMOsDatasetSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal',
                 buffer_size=160, use_gap=False):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)

        self.buffer_size = buffer_size
        self.samples = self.buffer_size * [None]
        self.frames_dict = None
        self.buffer_idx = 0
        self.use_gap = use_gap

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
            elif self.frame_sample_mode == 'continual':
                # Sample test frames and train frames in a continual manner, i.e. test_frame_ids and train_frame_ids are continual
                base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                idx_valid = False
                while not idx_valid:
                    train_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                             min_id=base_frame_id[
                                                        0] - self.max_gap - gap_increase,
                                             max_id=base_frame_id[
                                                        0] + self.max_gap + gap_increase - self.num_train_frames + 1)
                    train_frame_ids = [i for i in range(train_frame_start_idx[0],
                                                         train_frame_start_idx[0] + self.num_train_frames)]
                    if train_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[train_frame_ids].all()
                idx_valid = False
                while not idx_valid:
                    test_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                                                min_id=base_frame_id[
                                                                        0] - self.max_gap - gap_increase,
                                                                max_id=base_frame_id[
                                                                           0] + self.max_gap + gap_increase - self.num_test_frames + 1)
                    test_frame_ids = [i for i in range(test_frame_start_idx[0], test_frame_start_idx[0] + self.num_test_frames)]
                    if test_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[test_frame_ids].all()            
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        # print(f'train_ids:{train_frame_ids}, test_ids:{test_frame_ids}')
        train_frames, train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        test_frames, test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        train_anno = [{0: box} for box in train_anno['bbox']]
        test_anno = [{0: box} for box in test_anno['bbox']]

        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'dataset': "FE108",
                           'test_class': "Happy"})

        self.samples[self.buffer_idx] = None
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

        return self.processing(data)

    def _sample_training_data(self):
        num_tries = 100
        for i in range(num_tries):
            data = self._try_sample_training_data()
            if data is not None:
                return data
        raise RuntimeError()

    def _try_sample_training_data(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()
        is_mot_dataset = False

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
            elif self.frame_sample_mode == 'continual':
                # Sample test frames and train frames in a continual manner, i.e. test_frame_ids and train_frame_ids are continual
                base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                idx_valid = False
                while not idx_valid:
                    train_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                             min_id=base_frame_id[
                                                        0] - self.max_gap - gap_increase,
                                             max_id=base_frame_id[
                                                        0] + self.max_gap + gap_increase - self.num_train_frames + 1)
                    train_frame_ids = [i for i in range(train_frame_start_idx[0],
                                                         train_frame_start_idx[0] + self.num_train_frames)]
                    if train_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[train_frame_ids].all()
                idx_valid = False
                while not idx_valid:
                    test_frame_start_idx = self._sample_visible_ids(visible, num_ids=1,
                                                                min_id=base_frame_id[
                                                                        0] - self.max_gap - gap_increase,
                                                                max_id=base_frame_id[
                                                                           0] + self.max_gap + gap_increase - self.num_test_frames + 1)
                    test_frame_ids = [i for i in range(test_frame_start_idx[0], test_frame_start_idx[0] + self.num_test_frames)]
                    if test_frame_ids[-1] < len(visible) - 1:
                        idx_valid = visible[test_frame_ids].all()            
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        return train_frame_ids, test_frame_ids, seq_id

    def recompute_visiblity_for_train_frame_objects(self, visible_mot, train_frame_ids):
        cols = torch.where(visible_mot[train_frame_ids])[1]
        visible_any = torch.any(visible_mot[:, cols], dim=1)
        return visible_any
