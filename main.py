import torch.optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from classifier import GolemClassifier


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

    trainset = torchvision.datasets.CIFAR10(
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

def main():
    train_loader, test_loader, classes = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = models.resnet18()
    gc = GolemClassifier(resnet18, resnet18.fc.in_features)
    gc.to(device)


if __name__ == "__main__":
    main()
