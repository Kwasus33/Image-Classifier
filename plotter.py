import matplotlib.pyplot as plt


def plot_images(dataloader, pred_labels, class_names, grid_size=(4, 4)):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    plt.figure(figsize=(8, 8))
    for i in range(grid_size[0] * grid_size[1]):
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[pred_labels[i][0]])
    plt.show()
