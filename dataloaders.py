import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(test_run):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # transforms imgs to tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # new values: x' = (x−μ)/σ
            ),  # normalizes data - mean and std are set to 0.5 for R, G, B
        ]
    )

    batch_size = 32

    if test_run:
        trainset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=False, download=True, transform=transform
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root="./data/cifar100", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data/cifar100", train=False, download=True, transform=transform
        )

    # creates iterable set of batch sets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    class_names = (
        ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        if test_run
        else [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
    )

    return trainloader, testloader, class_names
