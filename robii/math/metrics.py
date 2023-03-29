from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
from numpy.linalg import norm



def psnr(estimated_image, true_image):
    estimated_image = estimated_image/(np.max(estimated_image) + 1e-6)
    true_image = true_image/(np.max(true_image) + 1e-6)

    err = norm(estimated_image - true_image)**2
    return 10 * np.log10(1 / err)

def ssim(estimated_image, true_image, *args, **kwargs):
    estimated_image = estimated_image/(np.max(estimated_image) + 1e-6)
    true_image = true_image/(np.max(true_image) + 1e-6)

    return structural_similarity(estimated_image, true_image, data_range=1, *args, **kwargs)

def nmse(estimated_image, true_image, reduce='norm'):
    if reduce == 'norm':
        return norm(estimated_image - true_image)**2 /  norm(true_image)**2
    if reduce == 'dim':
        return norm(estimated_image - true_image)**2 /  len(estimated_image.flatten())

def snr(estimated_image, true_image):
    return 10*np.log10(1/nmse(estimated_image, true_image))

def normalized_cross_correlation(X, Y):

    N = len(X.reshape(-1))

    sigmaX, sigmaY = np.sqrt(np.var(X)), np.sqrt(np.var(Y))
    muX, muY = np.mean(X), np.mean(Y)

    Xvec, Yvec = X.reshape(-1), Y.reshape(-1)

    return np.sum( [(Xi - muX)*(Yi - muY) for Xi,Yi in zip(Xvec,Yvec)])/(N*sigmaX*sigmaY)
    