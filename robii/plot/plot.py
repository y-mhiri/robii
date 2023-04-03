import matplotlib.pyplot as plt


def show_images(*args, cmap='Spectral_r', figsize=(10,10), save=False, out=None):

    nimages = len(args)

    if nimages == 1:
        plt.figure(figsize=figsize)
        plt.imshow(args[0][0], cmap=cmap)
        plt.title(args[0][1])

        if save:
            plt.savefig(f'{out}.png')
        else:
            plt.show()

        return True
       


    nrow, ncol = 1,nimages

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    for ii, image in enumerate(args):

        axes[0,ii].imshow(image[0], cmap=cmap)
        axes[0,ii].set_title(image[1])

    if save:
        plt.savefig(f'{out}.png')
    else:
        plt.show()

    return True