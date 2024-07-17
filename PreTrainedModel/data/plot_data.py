import matplotlib.pyplot as plt
import numpy as np


def PlotLoader(train_loader, val_loader, test_loader):
    fig, axs = plt.subplots(nrows=1, ncols=3)
    train_iter = iter(train_loader)
    train_images, _ = next(train_iter)
    train_out = train_images[0].numpy()
    axs[0].imshow(np.transpose(np.clip(a=train_out, a_min=0, a_max=255), (1, 2, 0)))
    axs[0].set_title('Train Image')

    val_iter = iter(val_loader)
    val_images, _ = next(val_iter)
    val_out = val_images[0].numpy()
    axs[1].imshow(np.transpose(np.clip(a=val_out, a_min=0, a_max=255), (1, 2, 0)))
    axs[1].set_title('Validation Image')

    test_iter = iter(test_loader)
    test_images, _ = next(test_iter)
    test_out = test_images[0].numpy()
    axs[2].imshow(np.transpose(np.clip(a=test_out, a_min=0, a_max=255), (1, 2, 0)))
    axs[2].set_title('Test Image')

    return plt.show()