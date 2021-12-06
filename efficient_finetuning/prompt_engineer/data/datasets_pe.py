from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
from torchvision import datasets, transforms, models
import torch

"""
Datasets that have been tested for the prompt engineering project
"""

class Template:
    def __init__(
        self,
        root=None,
    ):

        if root == None:
            root = "/nethome/bdevnani3/raid/data/"
        self.name = "Not Defined"

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

    def get_train_dataset(self, transform_fn):
        raise NotImplementedError

class Cifar10Clip(Template):
    def __init__(self, root=None):

        super().__init__(root)
        self.name = "CIFAR10"
        self.phrase = "This is a photo of a {}."
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
        
    def get_train_dataset(self, transform_fn=None):
        
        if transform_fn is None:
            transform_fn = self.train_transform
        
        train_dataset = CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )
        
        self.classes = train_dataset.classes
            
        return train_dataset

    def get_test_dataset(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform
        
        train_dataset = CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=transform_fn,
        )
        
        self.classes = train_dataset.classes
            
        return train_dataset

class Cifar100Clip(Template):
    def __init__(self, root=None):

        super().__init__(root)
        self.name = "CIFAR100"
        self.phrase = "This is a photo of a {}."
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
        
    def get_train_dataset(self, transform_fn=None):
        
        if transform_fn is None:
            transform_fn = self.train_transform
        
        train_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )
        
        self.classes = train_dataset.classes
            
        return train_dataset

    def get_test_dataset(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform
        
        train_dataset = CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=transform_fn,
        )
        
        self.classes = train_dataset.classes
            
        return train_dataset


class Flowers102Clip(Template):
    def __init__(self,root=None):
        super().__init__(root)
        self.name = "Flowers102"
        self.phrase = "A {} with petals."
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

    def get_train_dataset(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "flower_data/train", transform=transform_fn
        )
        return train_dataset

    def get_test_dataset(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "flower_data/test", transform=transform_fn
        )

        return test_dataset


class ImageNetClip(Template):
    def __init__(self, root=None):

        super().__init__(root)
        self.name = "ImageNet"
        self.phrase = "This is a photo of a {}."
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
        
    def get_train_dataset(self, transform_fn=None):
        
        if transform_fn is None:
            transform_fn = self.train_transform
        
        train_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )
        
        self.classes = train_dataset.classes
            
        return train_dataset

    def get_test_dataset(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform
        
        train_dataset = CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=transform_fn,
        )
        
        self.classes = train_dataset.classes
            
        return train_dataset