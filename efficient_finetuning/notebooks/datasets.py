from torchvision.datasets import CIFAR100
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import os
import torch
import numpy as np


class ClipExptDataset:
    def __init__(self, num_workers, batch_size):
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
        self.root = os.path.expanduser("/nethome/bdevnani3/raid/data/")

    def get_train_loaders(self, transform_fn):
        raise NotImplementedError

    def get_test_loader(self, transform_fn):
        raise NotImplementedError


##################################
############ DATASETS ############
##################################


class Cifar100(ClipExptDataset):
    def __init__(self, num_workers, batch_size):

        super().__init__(num_workers, batch_size)
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

    def get_train_loaders(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )

        valid_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
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

    wordnet_synonyms = {
        "pink primrose": [],
        "globe thistle": ["echinops", "herb", "globe thistle"],
        "blanket flower": [
            "indian blanket",
            "blanket flower",
            "gaillardia",
            "gaillardia pulchella",
            "fire-wheel",
            "fire wheel",
        ],
        "trumpet creeper": [
            "bignoniaceae",
            "campsis radicans",
            "trumpet vine",
            "trumpet creeper",
        ],
        "blackberry lily": [],
        "snapdragon": ["flower", "antirrhinum", "snapdragon"],
        "colt's foot": [],
        "king protea": [
            "honeypot",
            "protea cynaroides",
            "protea",
            "genus protea",
            "king protea",
        ],
        "spear thistle": [
            "cirsium lanceolatum",
            "bull thistle",
            "spear thistle",
            "boar thistle",
            "cirsium vulgare",
            "plume thistle",
        ],
        "yellow iris": [
            "yellow water flag",
            "yellow iris",
            "iris",
            "iris pseudacorus",
            "yellow flag",
        ],
        "globe-flower": [],
        "purple coneflower": [],
        "peruvian lily": [
            "alstroemeria",
            "alstroemeria pelegrina",
            "peruvian lily",
            "genus alstroemeria",
            "lily of the incas",
        ],
        "balloon flower": [
            "balloon flower",
            "penstemon palmeri",
            "scented penstemon",
            "penstemon",
            "wildflower",
        ],
        "hard-leaved pocket orchid": [],
        "giant white arum lily": [],
        "fire lily": [],
        "pincushion flower": [
            "sweet scabious",
            "scabious",
            "mournful widow",
            "scabiosa atropurpurea",
            "pincushion flower",
        ],
        "fritillary": [
            "bulbous plant",
            "checkered lily",
            "fritillary",
            "fritillaria",
        ],
        "red ginger": ["alpinia", "ginger", "red ginger", "alpinia purpurata"],
        "grape hyacinth": ["liliaceous plant", "grape hyacinth", "muscari"],
        "corn poppy": [
            "papaver",
            "field poppy",
            "corn poppy",
            "flanders poppy",
            "papaver rhoeas",
            "poppy",
        ],
        "prince of wales feathers": [],
        "stemless gentian": [],
        "artichoke": [
            "cynara scolymus",
            "cynara",
            "vegetable",
            "artichoke",
            "artichoke plant",
            "globe artichoke",
        ],
        "canterbury bells": [],
        "sweet william": ["dianthus barbatus", "sweet william", "pink"],
        "carnation": [
            "clove pink",
            "carnation",
            "dianthus caryophyllus",
            "gillyflower",
            "pink",
        ],
        "garden phlox": [],
        "love in the mist": [],
        "mexican aster": [],
        "alpine sea holly": [],
        "ruby-lipped cattleya": [],
        "cape flower": [],
        "great masterwort": [],
        "siam tulip": [],
        "sweet pea": [
            "lathyrus",
            "lathyrus odoratus",
            "sweet pea",
            "vine",
            "sweetpea",
        ],
        "lenten rose": [
            "lenten rose",
            "black hellebore",
            "hellebore",
            "helleborus orientalis",
        ],
        "barbeton daisy": [],
        "daffodil": ["narcissus", "narcissus pseudonarcissus", "daffodil"],
        "sword lily": [
            "genus gladiolus",
            "sword lily",
            "gladiolus",
            "glad",
            "gladiola",
            "iridaceous plant",
        ],
        "poinsettia": [
            "painted leaf",
            "euphorbia",
            "lobster plant",
            "spurge",
            "poinsettia",
            "christmas flower",
            "mexican flameleaf",
            "christmas star",
            "euphorbia pulcherrima",
        ],
        "bolero deep blue": [],
        "wallflower": ["flower", "erysimum", "wallflower"],
        "marigold": ["marigold", "flower", "tageteste"],
        "buttercup": [
            "butter-flower",
            "herb",
            "crowfoot",
            "butterflower",
            "ranunculus",
            "goldcup",
            "kingcup",
            "buttercup",
        ],
        "oxeye daisy": [
            "oxeye daisy",
            "leucanthemum",
            "composite",
            "chrysanthemum maximum",
            "leucanthemum maximum",
        ],
        "english marigold": [],
        "common dandelion": [
            "taraxacum officinale",
            "common dandelion",
            "dandelion",
            "taraxacum ruderalia",
        ],
        "petunia": ["flower", "genus petunia", "petunia"],
        "wild pansy": [
            "pink of my john",
            "wild pansy",
            "love-in-idleness",
            "viola",
            "heartsease",
            "johnny-jump-up",
            "viola tricolor",
        ],
        "primula": ["primula", "primrose", "herb", "genus primula"],
        "sunflower": ["flower", "genus helianthus", "sunflower", "helianthus"],
        "pelargonium": [
            "geraniaceae",
            "rosid dicot genus",
            "pelargonium",
            "genus pelargonium",
        ],
        "bishop of llandaff": [],
        "gaura": [],
        "geranium": ["geraniaceae", "herb", "geranium"],
        "orange dahlia": [],
        "tiger lily": [
            "tiger lily",
            "lily",
            "devil lily",
            "kentan",
            "lilium lancifolium",
        ],
        "pink-yellow dahlia": [],
        "cautleya spicata": [],
        "japanese anemone": [],
        "black-eyed susan": [
            "vine",
            "black-eyed susan vine",
            "thunbergia",
            "black-eyed susan",
            "thunbergia alata",
        ],
        "silverbush": [
            "anthyllis",
            "jupiter's beard",
            "anthyllis barba-jovis",
            "silverbush",
            "shrub",
            "silver-bush",
        ],
        "californian poppy": [],
        "osteospermum": [],
        "spring crocus": [],
        "bearded iris": ["bearded iris", "iris", "genus iris"],
        "windflower": ["flower", "genus anemone", "anemone", "windflower"],
        "moon orchid": [],
        "tree poppy": ["shrub", "bush poppy", "dendromecon", "tree poppy"],
        "gazania": ["flower", "gazania", "genus gazania"],
        "azalea": ["subgenus azalea", "azalea", "rhododendron"],
        "water lily": ["water lily", "aquatic plant", "nymphaeaceae"],
        "rose": ["shrub", "rose", "rosebush", "rosa"],
        "thorn apple": ["thorn apple", "datura", "shrub"],
        "morning glory": ["vine", "ipomoea", "morning glory"],
        "passion flower": [],
        "lotus": [
            "sacred lotus",
            "lotus",
            "water lily",
            "nelumbo nucifera",
            "indian lotus",
        ],
        "toad lily": ["montia chamissoi", "toad lily", "indian lettuce"],
        "bird of paradise": [
            "poinciana",
            "caesalpinia gilliesii",
            "bird of paradise",
            "flowering shrub",
            "poinciana gilliesii",
            "caesalpinia",
        ],
        "anthurium": [
            "houseplant",
            "genus anthurium",
            "tail-flower",
            "tailflower",
            "anthurium",
        ],
        "frangipani": ["shrub", "plumeria", "frangipanni", "frangipani"],
        "clematis": ["clematis", "vine", "genus clematis", "climber"],
        "hibiscus": ["genus hibiscus", "hibiscus", "mallow"],
        "columbine": [
            "aquilege",
            "flower",
            "columbine",
            "aquilegia",
            "genus aquilegia",
        ],
        "desert-rose": [],
        "tree mallow": [
            "lavatera arborea",
            "lavatera",
            "tree mallow",
            "velvet-leaf",
            "shrub",
            "velvetleaf",
        ],
        "magnolia": ["magnolia", "bark"],
        "cyclamen": [
            "flower",
            "cyclamen",
            "cyclamen purpurascens",
            "genus cyclamen",
        ],
        "watercress": ["cruciferae", "watercress", "cress"],
        "monkshood": [
            "helmetflower",
            "aconite",
            "monkshood",
            "aconitum napellus",
            "helmet flower",
        ],
        "canna lily": ["canna lily", "canna", "canna generalis"],
        "hippeastrum": [
            "hippeastrum",
            "genus hippeastrum",
            "amaryllis",
            "hippeastrum puniceum",
        ],
        "bee balm": ["bee balm", "monarda", "beebalm", "monarda fistulosa"],
        "ball moss": [],
        "foxglove": ["foxglove", "digitalis", "herb", "genus digitalis"],
        "bougainvillea": ["genus bougainvillea", "vine", "bougainvillea"],
        "camellia": ["shrub", "camellia", "camelia", "genus camellia"],
        "mallow": ["shrub", "malvaceae", "mallow"],
        "mexican petunia": [],
        "bromelia": ["bromeliaceae", "monocot genus", "bromelia"],
    }

    def __init__(self, num_workers, batch_size):
        super().__init__(num_workers, batch_size)
        self.name = "Flowers102"
        self.classes = [
            "pink primrose",
            "globe thistle",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "hard-leaved pocket orchid",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "canterbury bells",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "sweet pea",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "english marigold",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "tiger lily",
            "pink-yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "moon orchid",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "bird of paradise",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "monkshood",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
        ]

    def get_train_loaders(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "flower_data/train", transform=transform_fn
        )

        valid_dataset = datasets.ImageFolder(
            self.root + "flower_data/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
    def __init__(self, num_workers, batch_size):
        super().__init__(num_workers, batch_size)
        self.name = "OxfordPets"
        self.classes = [
            "Abyssinian",
            "Bengal",
            "Birman",
            "Bombay",
            "British",
            "Egyptian",
            "Maine",
            "Persian",
            "Ragdoll",
            "Russian",
            "Siamese",
            "Sphynx",
            "american",
            "basset",
            "beagle",
            "boxer",
            "chihuahua",
            "english",
            "german",
            "great",
            "havanese",
            "japanese",
            "keeshond",
            "leonberger",
            "miniature",
            "newfoundland",
            "pomeranian",
            "pug",
            "saint",
            "samoyed",
            "scottish",
            "shiba",
            "staffordshire",
            "wheaten",
            "yorkshire",
        ]

    def get_train_loaders(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/train", transform=transform_fn
        )

        valid_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
            self.root + "oxford_pets/test", transform=transform_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return test_loader
