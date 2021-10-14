from PIL.Image import NONE
from numpy.core.arrayprint import _none_or_positive_arg
from numpy.core.fromnumeric import _nonzero_dispatcher
from torchvision.datasets import CIFAR100
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import os
import torch
import numpy as np
import random


class ClipExptDataset:
    def __init__(
        self,
        num_workers,
        batch_size,
        root=None,
    ):

        if root == None:
            root = "/nethome/bdevnani3/raid/data/"
        self.name = "Not Defined"

        self.train_loader = None
        self.test_loader = None
        self.validate_loader = None

        self.clip_train_loader = None
        self.clip_test_loader = None
        self.clip_validate_loader = None

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = 0.1  # 10% of training data to be used as validation split
        self.classes = None

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Where the datasets are saved, override if location changes
        # self.root = os.path.expanduser("/nethome/bdevnani3/raid/data/")
        # self.root = os.path.expanduser('/usr0/home/gis/research/vis_lang/data/')
        self.root = root

    def get_train_loaders(self, transform_fn):
        raise NotImplementedError

    def get_test_loader(self, transform_fn):
        raise NotImplementedError


##################################
############ DATASETS ############
##################################


class Cifar100(ClipExptDataset):
    def __init__(self, num_workers, batch_size, root=None):

        super().__init__(num_workers, batch_size, root)
        self.name = "CIFAR100"
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
        )  # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
        )

    def get_train_loaders(self, transform_fn=None, num_elements_per_class=-1, clip_embedding=False, shuffle=True):

        if transform_fn is None:
            transform_fn = self.train_transform
            
        train_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )
        print(len(train_dataset))

        if num_elements_per_class >=0:
            print("HERE")
            train_dataset = get_n_items_per_class(train_dataset, num_elements_per_class)
            print(len(train_dataset))


        valid_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))
        random.shuffle(indices)

        train_idx, valid_idx = indices[:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
        )

        # Little hacky - need to improve
        if self.classes == None:
            self.classes = train_dataset.classes

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=transform_fn,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if self.classes == None:
            self.classes = test_dataset.classes

        return test_loader


##################################


class Flowers102(ClipExptDataset):
    def __init__(self, num_workers, batch_size, root=None):
        super().__init__(num_workers, batch_size, root)
        self.name = "Flowers102"
        self.class_label_mapping = {
            "pink primrose": 0,
            "globe thistle": 1,
            "blanket flower": 2,
            "trumpet creeper": 3,
            "blackberry lily": 4,
            "snapdragon": 5,
            "colt's foot": 6,
            "king protea": 7,
            "spear thistle": 8,
            "yellow iris": 9,
            "globe-flower": 10,
            "purple coneflower": 11,
            "peruvian lily": 12,
            "balloon flower": 13,
            "hard-leaved pocket orchid": 14,
            "giant white arum lily": 15,
            "fire lily": 16,
            "pincushion flower": 17,
            "fritillary": 18,
            "red ginger": 19,
            "grape hyacinth": 20,
            "corn poppy": 21,
            "prince of wales feathers": 22,
            "stemless gentian": 23,
            "artichoke": 24,
            "canterbury bells": 25,
            "sweet william": 26,
            "carnation": 27,
            "garden phlox": 28,
            "love in the mist": 29,
            "mexican aster": 30,
            "alpine sea holly": 31,
            "ruby-lipped cattleya": 32,
            "cape flower": 33,
            "great masterwort": 34,
            "siam tulip": 35,
            "sweet pea": 36,
            "lenten rose": 37,
            "barbeton daisy": 38,
            "daffodil": 39,
            "sword lily": 40,
            "poinsettia": 41,
            "bolero deep blue": 42,
            "wallflower": 43,
            "marigold": 44,
            "buttercup": 45,
            "oxeye daisy": 46,
            "english marigold": 47,
            "common dandelion": 48,
            "petunia": 49,
            "wild pansy": 50,
            "primula": 51,
            "sunflower": 52,
            "pelargonium": 53,
            "bishop of llandaff": 54,
            "gaura": 55,
            "geranium": 56,
            "orange dahlia": 57,
            "tiger lily": 58,
            "pink-yellow dahlia": 59,
            "cautleya spicata": 60,
            "japanese anemone": 61,
            "black-eyed susan": 62,
            "silverbush": 63,
            "californian poppy": 64,
            "osteospermum": 65,
            "spring crocus": 66,
            "bearded iris": 67,
            "windflower": 68,
            "moon orchid": 69,
            "tree poppy": 70,
            "gazania": 71,
            "azalea": 72,
            "water lily": 73,
            "rose": 74,
            "thorn apple": 75,
            "morning glory": 76,
            "passion flower": 77,
            "lotus": 78,
            "toad lily": 79,
            "bird of paradise": 80,
            "anthurium": 81,
            "frangipani": 82,
            "clematis": 83,
            "hibiscus": 84,
            "columbine": 85,
            "desert-rose": 86,
            "tree mallow": 87,
            "magnolia": 88,
            "cyclamen": 89,
            "watercress": 90,
            "monkshood": 91,
            "canna lily": 92,
            "hippeastrum": 93,
            "bee balm": 94,
            "ball moss": 95,
            "foxglove": 96,
            "bougainvillea": 97,
            "camellia": 98,
            "mallow": 99,
            "mexican petunia": 100,
            "bromelia": 101,
        }
        self.classes = list(self.class_label_mapping.keys())

    def get_train_loaders(self, transform_fn=None, num_elements_per_class=-1, shuffle=True):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "flower_data/train", transform=transform_fn
        )

        if num_elements_per_class >=0:
            train_dataset = get_n_items_per_class(train_dataset, num_elements_per_class)

        valid_dataset = datasets.ImageFolder(
            self.root + "flower_data/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "flower_data/test", transform=transform_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return test_loader


##################################


class OxfordPets(ClipExptDataset):
    def __init__(self, num_workers, batch_size, root=None):
        super().__init__(num_workers, batch_size, root)
        self.name = "OxfordPets"
        self.classes = [
            "Abyssinian",
            "Bengal",
            "Birman",
            "Bombay_cat",  # TODO: change this back to Bombay. Bombay_cat is something I added.
            "British_Shorthair",
            "Egyptian_Mau",
            "Maine_Coon",
            "Persian",
            "Ragdoll",
            "Russian_Blue",
            "Siamese",
            "Sphynx",
            "american_bulldog",
            "american_pit_bull_terrier",
            "basset_hound",
            "beagle",
            "boxer",
            "chihuahua",
            "english_cocker_spaniel",
            "english_setter",
            "german_shorthaired",
            "great_pyrenees",
            "havanese",
            "japanese_chin",
            "keeshond",
            "leonberger",
            "miniature_pinscher",
            "newfoundland",
            "pomeranian",
            "pug",
            "saint_bernard",
            "samoyed",
            "scottish_terrier",
            "shiba_inu",
            "staffordshire_bull_terrier",
            "wheaten_terrier",
            "yorkshire_terrier",
        ]

    def get_train_loaders(self, transform_fn=None, num_elements_per_class=-1, shuffle=True):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/train", transform=transform_fn
        )

        if num_elements_per_class >=0:
            train_dataset = get_n_items_per_class(train_dataset, num_elements_per_class)

        valid_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle
        )

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/test", transform=transform_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return test_loader



class SmallFlowers102(ClipExptDataset):

    def __init__(self, num_workers, batch_size, root=None):
        super().__init__(num_workers, batch_size, root)
        self.name = "SmallFlowers102"
        self.class_label_mapping = {
             'pelargonium': 53,
             'love in the mist': 29,
             'silverbush': 63,
             'great masterwort': 34,
             'bolero deep blue': 42,
             'hard-leaved pocket orchid': 14,
             'petunia': 49,
             'sword lily': 40,
             'wallflower': 43,
             'bougainvillea': 97,}

        
        self.classes = list(self.class_label_mapping.keys())
        self.new_labels = {v:i for i,v in enumerate(list(self.class_label_mapping.values()))}

    def get_train_loaders(self, transform_fn=None, num_elements_per_class=-1):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "flower_data/train", transform=transform_fn
        )

        train_dataset.target_transform = lambda id: self.new_labels[id]
        
        train_indices = np.argwhere(np.isin(np.array(train_dataset.targets),list(self.class_label_mapping.values()))).flatten()
        
        small_train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        small_train_dataset.targets = np.array(train_dataset.targets)[train_indices]

        if num_elements_per_class >=0:
            train_dataset = get_n_items_per_class(small_train_dataset, num_elements_per_class)

        valid_dataset = datasets.ImageFolder(
            self.root + "flower_data/valid", transform=transform_fn
        )

        valid_dataset.target_transform = lambda id: self.new_labels[id]
        valid_indices = np.argwhere(np.isin(np.array(valid_dataset.targets),list(self.class_label_mapping.values()))).flatten()

        small_valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)
        
        train_loader = torch.utils.data.DataLoader(
            small_train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            small_valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "flower_data/test", transform=transform_fn
        )
        
        test_dataset.target_transform = lambda id: self.new_labels[id]

        test_indices = np.argwhere(np.isin(np.array(test_dataset.targets),list(self.class_label_mapping.values()))).flatten()

        small_test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

        test_loader = torch.utils.data.DataLoader(
            small_test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return test_loader


class Food101(ClipExptDataset):
    def __init__(self, num_workers, batch_size, root=None):
        super().__init__(num_workers, batch_size, root)
        self.name = "Food101"

    def get_train_loaders(self, transform_fn=None, num_elements_per_class=-1, shuffle=True):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "food101/train", transform=transform_fn
        )
        clss = train_dataset.classes

        if num_elements_per_class >=0:
            train_dataset = get_n_items_per_class(train_dataset, num_elements_per_class)

        valid_dataset = datasets.ImageFolder(
            self.root + "food101/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle
        )

        if self.classes == None:
            self.classes = clss

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "food101/test", transform=transform_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return test_loader

def get_n_items_per_class(dataset, n):

    final_indices = []

    for label in list(set(dataset.targets)):
        class_inds = random.sample(list(np.argwhere(np.array(dataset.targets)==label).flatten()),n)
        final_indices.extend(class_inds)

    return torch.utils.data.Subset(dataset, final_indices)
