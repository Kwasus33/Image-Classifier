import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # transforms imgs to tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # new values: x' = (x−μ)/σ
            ),  # normalizes data - mean and std are set to 0.5 for R, G, B
        ]
    )

    batch_size = 32

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    # creates iterable set of batch sets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, testloader, classes
