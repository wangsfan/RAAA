import os
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Optional, Callable, Dict, Tuple
import warnings

import lmdb
import math
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

from iirc.definitions import NO_LABEL_PLACEHOLDER
from iirc.lifelong_dataset.torch_dataset import Dataset
from lifelong_methods.utils import SubsetSampler


class BufferBase(ABC, data.Dataset):
    """
    A buffer that saves memories from current task to replay them during later tasks

    Args:
        config (Dict): A dictionary that has the following key value pairs:
            n_memories_per_class (int): Number of memories/samples to save per class, set to -1 to use total_n_mems
            total_n_mems (int): The total number of memories to save (mutually exclusive with n_memories_per_class)
            max_mems_pool_per_class (int): The pool size per class to sample randomly the memories from which the buffer
                chooses what memories to keep, set to -1 to choose memories from all the class samples
        buffer_dir (Optional[str]): The directory where the buffer data will be kept (None for keeping the buffer data
            in memory) (default: None)
        map_size (int): Th estimated size of the buffer lmdb database, in bytes (defalt: 1e9)
        essential_transforms_fn (Optional[Callable[[Image.Image], torch.Tensor]]): A function that contains the
            essential transforms (for example, converting a pillow image to a tensor) that should be applied to each
            image. This function is applied only when the augmentation_transforms_fn is set to None (as in the case
            of a test set) or inside the disable_augmentations context (default: None)
        augmentation_transforms_fn: (Optional[Callable[[Image.Image], torch.Tensor]]): A function that contains the
            essential transforms (for example, converting a pillow image to a tensor) and augmentation transforms (for
            example, applying random cropping) that should be applied to each image. When this function is provided,
            essential_transforms_fn is not used except inside the disable_augmentations context (default: None)
     """

    def __init__(self,
                 args,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        self.max_mems_pool_size = args.max_mems_pool_per_class
        if args.n_memories_per_class > -1:
            self.n_mems_per_cla = args.n_memories_per_class
            self.total_n_mems = 0
            self.fixed_n_mems_per_cla = True
        else:
            self.total_n_mems = args.total_n_memories
            self.n_mems_per_cla = 0
            self.fixed_n_mems_per_cla = False

        self.essential_transforms_fn = essential_transforms_fn
        self.augmentation_transforms_fn = augmentation_transforms_fn
        if essential_transforms_fn is None:
            self.essential_transforms_fn = transforms.ToTensor()
        if augmentation_transforms_fn is None:
            self.augmentation_transforms_fn = self.essential_transforms_fn
        self._apply_augmentations = True

        self.seen_classes = []

        self.mem_class_x = OrderedDict()  # 该字典存放各类对应的buffer样本，其中key是类别名str，val是buffer中该类别对应的所有样本dict（encoded_image）

        self.map_size = map_size
        self._lmdb_env = None
        self._txn = None
        if buffer_dir is not None and (self.n_mems_per_cla > 0 or self.total_n_mems > 0):
            self.save_buffer_on_disk = True
            self.buffer_db_dir = os.path.join(buffer_dir, "buffer_data")
            self._lmdb_env = lmdb.Environment(self.buffer_db_dir, map_size=self.map_size, max_spare_txns=6)
        else:
            self.save_buffer_on_disk = False
            self.buffer_db_dir = None

        # attributes not to be saved and loaded
        self.non_savable_attributes = [
            "essential_transforms_fn", "augmentation_transforms_fn", "buffer_db_dir", "map_size", "_lmdb_env", "_txn",
            "non_savable_attributes"]

    def __len__(self):
        return sum([len(class_list) for class_list in list(self.mem_class_x.values())])

    def _is_lmdb_env_created(self) -> bool:
        result = self._lmdb_env is not None
        return result

    def buffer_state_dict(self) -> Dict:
        """
        This function returns a dict that contains the current state of the buffer

        Returns:
            Dict: a dictionary with all the attributes (key is attribute name) and their values, except the
            attributes in the self.non_savable_attributes
        """
        state_dict = {}
        for key in self.__dict__.keys():
            if key not in self.non_savable_attributes:
                state_dict[key] = self.__dict__[key]
        return state_dict

    def load_buffer_state_dict(self, state_dict: Dict) -> None:
        """
        This function loads the object attributes with the values in state_dict

        Args:
            state_dict (Dict): a dictionary with the attribute names as keys and their values
        """
        for key in state_dict.keys():
            if key not in self.non_savable_attributes:
                self.__dict__[key] = state_dict[key]

    def get_image_indices_by_class(self, class_name: str) -> np.ndarray:
        """
        get the indices of the samples of class "class_name"

        Args:
            class_name (str): The class name

        Returns:
            np.ndarray: The indices of the samples of class "class_name"
        """
        start_index = 0
        class_images_indices = []
        for class_ in self.mem_class_x:
            if class_ == class_name:
                class_images_indices = np.arange(start_index, start_index + len(self.mem_class_x[class_]))
            else:
                start_index += len(self.mem_class_x[class_])
        return class_images_indices

    def _encode_image(self, image: Image.Image, labels: Tuple[str, str]) -> Dict:
        return {
            "image": image.tobytes(),
            "labels": labels,
            "size": image.size,
            "mode": image.mode
        }

    def _decode_image(self, encoded_image) -> Tuple[Image.Image, str, str]:
        image = Image.frombytes(encoded_image["mode"], encoded_image["size"], encoded_image["image"])
        labels = encoded_image["labels"]
        return image, labels[0], labels[1]

    def _create_key(self, class_label, per_class_index) -> str:
        return f"{class_label}_{per_class_index:06}"

    def _get_index_from_key(self, key: str) -> int:
        return int(key.split("_")[-1])

    def begin_adding_samples_to_lmdb(self) -> None:
        """
        A function that needs to be called before adding samples to the buffer, in case of using an lmdb buffer, so that
            a transaction is created.
        """
        if self.save_buffer_on_disk:
            if not self._is_lmdb_env_created():
                self.reset_lmdb_database()
            self._txn = lmdb.Transaction(self._lmdb_env, write=True)

    def end_adding_samples_to_lmdb(self) -> None:
        """
        A function that needs to be called after adding samples to the buffer is done, in case of using an lmdb buffer,
            so that the transaction is committed.
        """
        if self.save_buffer_on_disk:
            self._txn.commit()
            self._txn = None

    def reset_lmdb_database(self) -> None:
        """
        A function that needs to be called after each epoch, in case of using an lmdb dataset, to close the environment
            and open a new one to kill active readers
        """
        if self.save_buffer_on_disk:
            if self._is_lmdb_env_created():
                self._lmdb_env.close()
            self._lmdb_env = lmdb.Environment(self.buffer_db_dir, map_size=self.map_size, max_spare_txns=6)

    def add_sample(self, class_label: str, image: Image.Image, labels: Tuple[str, str], rank: int = 0) -> None:
        """
        Add a sample to the buffer.

        Args:
            class_label (str): The class label of the image, and in case the image has multiple labels, the class label
                for which the sample should be associated with in the buffer
            image (Image.Image): The image to be added
            labels (Tuple[str, str]): The labels of the image (including the class_label), in case the image has only
                one label, provide the second label as NO_LABEL_PLACEHOLDER
            rank (int): The rank of the current gpu, in case of using multiple gpus
        """
        encoded_image = self._encode_image(image, labels)
        if self.save_buffer_on_disk:
            assert self._txn is not None, "call begin_adding_samples_to_lmdb before add_sample"
            if len(self.mem_class_x[class_label]) > 0:
                per_class_index = self._get_index_from_key(self.mem_class_x[class_label][-1]) + 1
            else:
                per_class_index = 0
            key = self._create_key(class_label, per_class_index)
            if key in self.mem_class_x[class_label]:
                raise ValueError(f"The {key} already exists in the buffer")
            self.mem_class_x[class_label].append(key)
            if rank == 0:
                self._txn.put(key.encode("ascii"), pickle.dumps(encoded_image))
        else:
            self.mem_class_x[class_label].append(encoded_image)

    def remove_samples(self, class_label: str, n: int) -> None:
        """
        Remove a number (n) of the samples associated with class "class_label".

        Args:
            class_label (str): The class label of which the sample is associated with in the buffer
            n (int): The number of samples to remove
        """
        last_index = len(self.mem_class_x[class_label]) - 1
        first_index = last_index - n
        for i in range(last_index, first_index, -1):
            if self.save_buffer_on_disk:
                key = self.mem_class_x[class_label][i]
                with self._lmdb_env.begin(write=True) as txn:
                    txn.delete(key.encode("ascii"))
                del self.mem_class_x[class_label][i]
            else:
                del self.mem_class_x[class_label][i]

    def _fetch_item(self, class_label, per_class_index) -> Tuple[Image.Image, str, str]:
        if self.save_buffer_on_disk:
            if not self._is_lmdb_env_created():
                self.reset_lmdb_database()
            key = self.mem_class_x[class_label][per_class_index]
            with self._lmdb_env.begin(write=False) as txn:
                encoded_image = pickle.loads(bytes(txn.get(key.encode("ascii"), default=None)))
        else:
            encoded_image = self.mem_class_x[class_label][per_class_index]
        assert encoded_image is not None, f"key {key} doesn't exist"
        image, label1, label2 = self._decode_image(encoded_image)
        return image, label1, label2

    def __getitem__(self, index) -> Tuple[torch.Tensor, str, str]:
        if index < 0:
            index += len(self)
        image = None
        label1 = NO_LABEL_PLACEHOLDER
        label2 = NO_LABEL_PLACEHOLDER
        per_class_index = index
        for class_ in self.mem_class_x:
            if per_class_index < len(self.mem_class_x[class_]):
                image, label1, label2 = self._fetch_item(class_, per_class_index)
                break
            else:
                per_class_index -= len(self.mem_class_x[class_])

        if self._apply_augmentations:
            image = self.augmentation_transforms_fn(image)
        else:
            image = self.essential_transforms_fn(image)

        return image, label1, label2

    @abstractmethod
    def _reduce_exemplar_set(self, **kwargs) -> None:
        """remove extra exemplars from the buffer (implement in the Buffer class in the method file)"""
        pass

    @abstractmethod
    def _construct_exemplar_set(self, task_data: Dataset, **kwargs) -> None:
        """update the buffer with the new task exemplars (implement in the Buffer class in the method file)"""
        pass

    def update_buffer_new_task(self, new_task_data: Dataset, args,
                               **kwargs) -> None:
        """
        Update the buffer by adding samples of classes of a new task, after removing samples associated with the older
            classes in case the buffer has a fixed size (self.fixed_n_mems_per_cla is set to False)

        Args:
            new_task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            **kwargs: arguments associated with each method
        """
        self.seen_classes = list(set(new_task_data.seen_classes) | set(self.seen_classes))
        for class_name in new_task_data.cur_task:
            self.mem_class_x[class_name] = []

        # if not self.fixed_n_mems_per_cla:
        #     self.n_mems_per_cla = math.ceil(self.total_n_mems / len(self.seen_classes))
        #     self._reduce_exemplar_set(dist_args=dist_args, **kwargs)

        self.begin_adding_samples_to_lmdb()
        self._construct_exemplar_set(new_task_data, args, **kwargs)
        self.end_adding_samples_to_lmdb()

    def _modify_exemplar_set(self, new_batch_data, **kwargs):
        """
        augment the buffer with some data for the present tasks, used for online setting where the task data keeps
            coming and the buffer should check if it needs to add or remove from the current task exemplars
        """
        raise NotImplementedError

    @contextmanager
    def disable_augmentations(self):
        """A context where only the essential transformations are applied"""
        cur_augmentations_state = self._apply_augmentations
        self._apply_augmentations = False
        try:
            yield
        finally:
            self._apply_augmentations = cur_augmentations_state


class Buffer(BufferBase):
    def __init__(self,
                 args,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(args, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)
        self.device = torch.device(args.device)

    def _reduce_exemplar_set(self, **kwargs) -> None:
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)


    # def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[dict] = None,
    #                             model: torch.nn.Module = None, batch_size=1, **kwargs):
    #     """
    #     Update the buffer with the new task samples using herding
    #
    #     Args:
    #         task_data (Dataset): The new task data
    #         dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
    #         rank of the device) (default: None)
    #         model (BaseMethod): The current method object to calculate the latent variables
    #         batch_size (int): The minibatch size
    #     """
    #     distributed = dist_args is not None
    #     if distributed:
    #         device = torch.device(f"cuda:{dist_args['gpu']}")
    #         rank = dist_args['rank']
    #     else:
    #         device = self.device
    #         rank = 0
    #     new_class_labels = task_data.cur_task
    #     model.eval()
    #
    #     with task_data.disable_augmentations(): # disable augmentations then enable them (if they were already enabled)
    #         with torch.no_grad():
    #             for class_label in new_class_labels:
    #                 class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
    #                 if distributed:
    #                     device = torch.device(f"cuda:{dist_args['gpu']}")
    #                     class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
    #                     dist.broadcast(class_data_indices_to_broadcast, 0)
    #                     class_data_indices = class_data_indices_to_broadcast.cpu().numpy()
    #                 sampler = SubsetSampler(class_data_indices)
    #                 class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler)
    #                 latent_vectors = []
    #                 for minibatch in class_loader:
    #                     images = minibatch[0].to(device)
    #                     output, out_latent = model.forward_net(images)
    #                     out_latent = out_latent.detach()
    #                     out_latent = F.normalize(out_latent, p=2, dim=-1)
    #                     latent_vectors.append(out_latent)
    #                 latent_vectors = torch.cat(latent_vectors, dim=0)
    #                 class_mean = torch.mean(latent_vectors, dim=0)
    #
    #                 chosen_exemplars_ind = []
    #                 exemplars_mean = torch.zeros_like(class_mean)
    #                 while len(chosen_exemplars_ind) < min(self.n_mems_per_cla, len(class_data_indices)):
    #                     potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(chosen_exemplars_ind) + latent_vectors) \
    #                                                / (len(chosen_exemplars_ind) + 1)
    #                     distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
    #                     shuffled_index = torch.argmin(distance).item()
    #                     exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
    #                     exemplar_index = class_data_indices[shuffled_index]
    #                     chosen_exemplars_ind.append(exemplar_index)
    #                     latent_vectors[shuffled_index, :] = float("inf")
    #
    #                 for image_index in chosen_exemplars_ind:
    #                     image, label1, label2 = task_data.get_item(image_index)
    #                     if label2 != NO_LABEL_PLACEHOLDER:
    #                         warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
    #                     self.add_sample(class_label, image, (label1, label2), rank=rank)

    def _construct_exemplar_set(self, task_data: Dataset, args,
                                model: torch.nn.Module = None, batch_size=1, device:torch.device=None, **kwargs):
        """
        Update the buffer with the new task samples using herding

        Args:
            task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            model (BaseMethod): The current method object to calculate the latent variables
            batch_size (int): The minibatch size
        """
        # distributed = dist_args is not None
        # if distributed:
        #     device = torch.device(f"cuda:{dist_args['gpu']}")
        #     rank = dist_args['rank']
        # else:
        #     device = self.device
        #     rank = 0
        rehearsal = args.rehearsal
        new_class_labels = task_data.cur_task
        model.eval()

        with task_data.disable_augmentations(): # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    # if distributed:
                    #     device = torch.device(f"cuda:{dist_args['gpu']}")
                    #     class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                    #     dist.broadcast(class_data_indices_to_broadcast, 0)
                    #     class_data_indices = class_data_indices_to_broadcast.cpu().numpy()

                    if rehearsal == "random":
                        indexes = np.random.choice(class_data_indices, size=self.n_mems_per_cla)

                    elif "closest" in rehearsal:
                        if rehearsal == 'closest_token':
                            handling = 'last'
                        else:
                            handling = 'all'

                        class_features = extract_features(task_data, class_data_indices, batch_size, device, model, handling)
                        class_features = class_features.cpu().numpy()

                        class_mean = np.mean(class_features, axis=0, keepdims=True)
                        distances = np.power(class_features - class_mean, 2).sum(-1)
                        class_closest_indexes = np.argsort(distances)

                        indexes = class_data_indices[class_closest_indexes[:self.n_mems_per_cla]]

                    elif "furthest" in rehearsal:
                        if rehearsal == 'furthest_token':
                            handling = 'last'
                        else:
                            handling = 'all'

                        class_features = extract_features(task_data, class_data_indices, batch_size, device, model, handling)
                        class_features = class_features.cpu().numpy()

                        class_mean = np.mean(class_features, axis=0, keepdims=True)
                        distances = np.power(class_features - class_mean, 2).sum(-1)
                        class_furthest_indexes = np.argsort(distances)[::-1]

                        indexes = class_data_indices[class_furthest_indexes[:self.n_mems_per_cla]]

                    elif "icarl":
                        if rehearsal == 'icarl_token':
                            handling = 'last'
                        else:
                            handling = 'all'

                        class_features = extract_features(task_data, class_data_indices, batch_size, device, model, handling)
                        class_features = class_features.cpu().numpy()

                        indexes = class_data_indices[icarl_selection(class_features, self.n_mems_per_cla)]

                    else:
                        raise ValueError(f"Unknown rehearsal method {rehearsal}!")

                    for image_index in indexes:
                        image, label1, label2 = task_data.get_item(image_index)
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2))


class TaskDataMergedWithBuffer(data.Dataset):
    """
    A torch dataset object that merges the task data and the buffer with the specified options

    Args:
        buffer (BufferBase): A buffer object that includes the memories from previous classes
        task_data (data.Dataset): A dataset object that contains the new task data
        buffer_sampling_multiplier (float): A multiplier for sampling from the buffer more/less times than the size
            of the buffer (for example a multiplier of 2 samples from the buffer (with replacement) twice its size per
            epoch, a multiplier of 1 ensures that all the buffer samples will be retrieved once")
    """

    def __init__(self, buffer: BufferBase, task_data: Dataset, buffer_sampling_multiplier: float = 1.0):
        self.buffer = buffer
        self.task_data = task_data
        self.num_samples = len(self.task_data) + len(self.buffer)
        self.seen_classes = list(set(task_data.seen_classes) | set(self.buffer.seen_classes))

        self.buffer_sampling_multiplier = buffer_sampling_multiplier
        self._buffer_sampling_array = self._get_buffer_index_sampling_array()

    def _get_buffer_index_sampling_array(self) -> np.ndarray:
        bf_len = len(self.buffer)
        multiplier = self.buffer_sampling_multiplier
        # if the multiplier is 1, this ensures that all the buffer samples will be retrieved once
        buffer_sampling_array = np.random.permutation(math.ceil(multiplier) * bf_len)
        buffer_sampling_array = buffer_sampling_array[:int(multiplier * bf_len)]
        buffer_sampling_array %= bf_len
        return buffer_sampling_array

    def __len__(self):
        """The number of samples, counting the length of the buffer after taking the buffer sampling multiplier into
        account"""
        tsk_data_len = len(self.task_data)
        bf_len = len(self.buffer)
        multiplier = self.buffer_sampling_multiplier
        return tsk_data_len + int(multiplier * bf_len)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str, str, bool]:
        if index < 0:
            index = len(self) + index
        tsk_data_len = len(self.task_data)
        # print("tsk_data_len : ", tsk_data_len)
        if index < tsk_data_len:
            # print("index : ", index)
            image, label_1, label_2 = self.task_data[index]
            in_buffer = False
        else:
            index -= tsk_data_len
            image, label_1, label_2 = self.buffer[self._buffer_sampling_array[index]]
            in_buffer = True
        assert label_1 in self.seen_classes + [NO_LABEL_PLACEHOLDER]
        assert label_2 in self.seen_classes + [NO_LABEL_PLACEHOLDER]
        return image, label_1, label_2, in_buffer


# def herd_samples(class_imgs, model, memory_per_class, rehearsal):
#     if rehearsal == "random":
#         indexes = []
#         for class_id in np.unique(y):
#             class_indexes = np.where(y == class_id)[0]
#             indexes.append(
#                 np.random.choice(class_indexes, size=memory_per_class)
#             )
#         indexes = np.concatenate(indexes)
#
#         return x[indexes], y[indexes], t[indexes]
#     elif "closest" in rehearsal:
#         if rehearsal == 'closest_token':
#             handling = 'last'
#         else:
#             handling = 'all'
#
#         features, targets = extract_features(dataset, model, handling)
#         indexes = []
#
#         for class_id in np.unique(y):
#             class_indexes = np.where(y == class_id)[0]
#             class_features = features[class_indexes]
#
#             class_mean = np.mean(class_features, axis=0, keepdims=True)
#             distances = np.power(class_features - class_mean, 2).sum(-1)
#             class_closest_indexes = np.argsort(distances)
#
#             indexes.append(
#                 class_indexes[class_closest_indexes[:memory_per_class]]
#             )
#
#         indexes = np.concatenate(indexes)
#         return x[indexes], y[indexes], t[indexes]
#     elif "furthest" in rehearsal:
#         if rehearsal == 'furthest_token':
#             handling = 'last'
#         else:
#             handling = 'all'
#
#         features, targets = extract_features(dataset, model, handling)
#         indexes = []
#
#         for class_id in np.unique(y):
#             class_indexes = np.where(y == class_id)[0]
#             class_features = features[class_indexes]
#
#             class_mean = np.mean(class_features, axis=0, keepdims=True)
#             distances = np.power(class_features - class_mean, 2).sum(-1)
#             class_furthest_indexes = np.argsort(distances)[::-1]
#
#             indexes.append(
#                 class_indexes[class_furthest_indexes[:memory_per_class]]
#             )
#
#         indexes = np.concatenate(indexes)
#         return x[indexes], y[indexes], t[indexes]
#     elif "icarl":
#         if rehearsal == 'icarl_token':
#             handling = 'last'
#         else:
#             handling = 'all'
#
#         features, targets = extract_features(dataset, model, handling)
#         indexes = []
#
#         for class_id in np.unique(y):
#             class_indexes = np.where(y == class_id)[0]
#             class_features = features[class_indexes]
#
#             indexes.append(
#                 class_indexes[icarl_selection(class_features, memory_per_class)]
#             )
#
#         indexes = np.concatenate(indexes)
#         return x[indexes], y[indexes], t[indexes]
#     else:
#         raise ValueError(f"Unknown rehearsal method {rehearsal}!")



def extract_features(task_data, class_data_indices, batch_size, device, model, ensemble_handling='last'):
    sampler = SubsetSampler(class_data_indices)
    class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=False, shuffle=False)

    features = []
    for minibatch in class_loader:
        images = minibatch[0].to(device)

        if hasattr(model, 'module'):
            feats, _, _ = model.module.forward_features(images)
        else:
            feats, _, _ = model.forward_features(images)

        if isinstance(feats, list):
            if ensemble_handling == 'last':
                feats = feats[-1]
            elif ensemble_handling == 'all':
                feats = torch.cat(feats, dim=1)
            else:
                raise NotImplementedError(f'Unknown handdling of multiple features {ensemble_handling}')
        elif len(feats.shape) == 3:  # joint tokens
            if ensemble_handling == 'last':
                feats = feats[-1]
            elif ensemble_handling == 'all':
                feats = feats.permute(1, 0, 2).view(len(images), -1)
            else:
                raise NotImplementedError(f'Unknown handdling of multiple features {ensemble_handling}')

        features.append(feats)

    features = torch.cat(features, dim=0)

    return features


def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def get_finetuning_dataset(dataset, buffer, finetuning='balanced', oversample_old=1, task_id=0):
    if finetuning == 'balanced':

        # if oversample_old > 1:
        #     old_indexes = np.where(t < task_id)[0]
        #     assert len(old_indexes) > 0
        #     new_indexes = np.where(t >= task_id)[0]
        #
        #     indexes = np.concatenate([
        #         np.repeat(old_indexes, oversample_old),
        #         new_indexes
        #     ])
        #     x, y, t = x[indexes], y[indexes], t[indexes]

        new_dataset = buffer
    elif finetuning in ('all', 'none'):
        new_dataset = dataset
    else:
        raise NotImplementedError(f'Unknown finetuning method {finetuning}')

    return new_dataset

