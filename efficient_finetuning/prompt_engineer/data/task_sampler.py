import random
from typing import List, Tuple

import torch
from torch.utils.data import Sampler, Dataset

import clip

class CLIPTaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self, dataset: Dataset, n_way: int, n_shot: int, n_query: int, n_tasks: int, clip_model, phrase="This is a photo of a {}", testing=False
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample/episodes
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.classes = dataset.classes
        self.phrase = phrase
        self.clip_model = clip_model
        self.testing = testing
        self.clip_model.eval()

        self.items_per_label = {}
        assert hasattr(
            dataset, "labels"
        ), "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        new_input_data  = []

        cache = {}
        
        for image, label in input_data:

            image_emb = self.clip_model.encode_image(image.unsqueeze(0).cuda())
            image_emb = image_emb/image_emb.norm(dim=-1, keepdim=True)

            #speeeed up
            if label in cache:
                class_embeddings = cache[label]
            else:
                if self.testing == True:
                    label = label - 92
                class_name = self.classes[label]
                class_name = class_name.replace("_", " ")
                text = clip.tokenize(self.phrase.format(class_name))
                class_embeddings = self.clip_model.encode_text(text.cuda())
                class_embeddings = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                cache[label] = class_embeddings

            final_input = torch.cat((image_emb, class_embeddings),dim=1)
            new_input_data.append((final_input.squeeze(0),label))

        true_class_ids = list({x[1] for x in new_input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in new_input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in new_input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].clone().reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].clone().reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        return (
            support_images.float(),
            support_labels,
            query_images.float(),
            query_labels,
            true_class_ids,
        )

    def episodic_collate_fn_test(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        new_input_data  = []

        cache = {}

        print(len(input_data))
        
        for image, label in input_data:

            image_emb = self.clip_model.encode_image(image.unsqueeze(0).cuda())
            image_emb = image_emb/image_emb.norm(dim=-1, keepdim=True)

            #speeeed up
            if label in cache:
                class_embeddings = cache[label]
            else:
                if self.testing == True:
                    label = label - 92
                class_name = self.classes[label]
                class_name = class_name.replace("_", " ")
                text = clip.tokenize(self.phrase.format(class_name))
                class_embeddings = self.clip_model.encode_text(text.cuda())
                class_embeddings = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                cache[label] = class_embeddings

            final_input = torch.cat((image_emb, class_embeddings),dim=1)
            new_input_data.append((final_input.squeeze(0),label))

        true_class_ids = list({x[1] for x in new_input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in new_input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in new_input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].clone().reshape(
            (-1, *all_images.shape[2:])
        )
        support_labels = all_labels[:, : self.n_shot].flatten()
        return (
            support_images.float(),
            support_labels,
            true_class_ids,
        )