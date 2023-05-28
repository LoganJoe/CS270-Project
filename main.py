import cv2
import numpy as np

import pyramids
from pyramids import Pyramids
import scipy.io as scio
from skimage import exposure

def DFT(img):
    return np.fft.fftshift(np.fft.fft2(img))


def IDFT(img):
    return np.fft.ifft2(np.fft.ifftshift(img))


def visualize_DFT(img, display=True):
    img_fft = DFT(img)
    img_fft = np.log(np.abs(img_fft))
    img_fft = (img_fft - np.min(img_fft)) / (np.max(img_fft) - np.min(img_fft))
    if display:
        cv2.imshow('DFT', img_fft)
        cv2.waitKey(0)
        cv2.imwrite("log_dft.jpg", img_fft * 255)
    return img_fft


def get_inner_radius(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edge = cv2.Canny((img*255).astype(np.uint8), 40, 89)
    avg_r = 0
    for x in range(edge.shape[0]):
        for y in range(edge.shape[1]):
            if edge[x][y] == 255:
                if avg_r < np.sqrt((x - edge.shape[0] // 2) ** 2 + (y - edge.shape[1] // 2) ** 2):
                    avg_r = np.sqrt((x - edge.shape[0] // 2) ** 2 + (y - edge.shape[1] // 2) ** 2)
    r = avg_r
    cv2.imwrite('PSF_edge.png', edge)
    # r = 79 / 2
    return (3.83 * img.shape[0]) / (2 * np.pi * r)


def get_motion_blur_params(img):
    img = cv2.GaussianBlur(img, (5, 5), 0.5)
    edge = cv2.Canny((img*255).astype(np.uint8), 70, 180)
    cv2.imwrite('MB_edge.png', edge)
    a = -0.021
    b = 97 / 57 * 0.021
    T = 500
    return a, b, T


def get_PSF_kernel(size, radius):
    """
    :param size: size of the kernel
    :param radius: radius of the inner circle
    :return: PSF kernel in frequency domain
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i - size // 2) ** 2 + (j - size // 2) ** 2 <= radius ** 2:
                kernel[i][j] = 1
    return DFT(kernel)


def get_MotionBlur_kernel(size, a, b, T):
    """
    :param size: size of the kernel
    :param a: proportional to extent of motion blur in vertical direction
    :param b: proportional to extent of motion blur in horizontal direction
    :param T: exposure time
    :return: MotionBlur kernel in frequency domain
    """
    kernel = np.zeros((size, size)).astype(np.complex128)
    for u in range(-size // 2, size // 2):
        for v in range(-size // 2, size // 2):
            param = np.pi * (u * a + v * b)
            if param == 0:
                kernel[u + size // 2][v + size // 2] = T * np.exp(-1.0j * param)
            else:
                kernel[u + size // 2][v + size // 2] = T / param * np.sin(param) * np.exp(-1.0j * param)
    return kernel


def restore_Trump(img, R=None, K=50):
    """
    :param img: The image of Trump with 3 channels
    :param R: manually measured radius of the inner circle of the PSF kernel
    :param K: The inverse of SNR i.e. noise-to-signal-ratio
    :return: The restored image of Trump
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = visualize_DFT(img_gray, display=False)
    R = get_inner_radius(img_fft) if R is None else R
    PSF_kernel = get_PSF_kernel(img_fft.shape[0], R)
    restored_img = Wiener_filter(img, PSF_kernel, K)
    restored_img = np.fft.ifftshift(restored_img, axes=(0, 1))
    return restored_img


def restore_Biden(img, a=None, b=None, T=None, K=300):
    """
    :param img: image of Biden with 3 channels
    :param a: proportional to extent of motion blur in vertical direction
    :param b: proportional to extent of motion blur in horizontal direction
    :param T: exposure time
    :param K: inverse of SNR i.e. noise-to-signal-ratio
    :return: restored image of Biden
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = visualize_DFT(img_gray, display=False)
    a, b, T = get_motion_blur_params(img_fft) if a is None or b is None or T is None else (a, b, T)
    MB_kernel = get_MotionBlur_kernel(img_gray.shape[0], a, b, T)
    restored_img = Wiener_filter(img, MB_kernel, K)
    return restored_img


def Wiener_filter(img, kernel, K):
    """
    :param img: image to be filtered in spatial domain
    :param kernel: kernel in frequency domain
    :param K: inverse of SNR i.e. noise-to-signal-ratio
    :return: restored image in spatial domain
    """
    if len(img.shape) == 3:
        res = np.zeros(img.shape).astype(np.uint8)
        for i in range(img.shape[-1]):
            res[..., i] = Wiener_filter(img[..., i], kernel, K)
        return res
    else:
        img_fft = DFT(img)
        res_fft = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        res = np.abs(IDFT(img_fft * res_fft))
        res = ((res - np.min(res)) / (np.max(res) - np.min(res)) * 255).astype(np.uint8)
        return res


if __name__ == "__main__":
    Biden = cv2.imread('images/man1.jpg').astype(np.float32) / 255
    Trump = cv2.imread('images/man2.jpg').astype(np.float32) / 255
    mask = scio.loadmat('images/mask.mat')['mask'].astype(np.float32)
    mask /= 255
    mask = 1 - mask
    res_Trump = restore_Trump(Trump)
    res_Biden = restore_Biden(Biden)
    cv2.imwrite('restored_Trump.jpg', res_Trump)
    cv2.imwrite('restored_Biden.jpg', res_Biden)
    # pyramids = Pyramids()
    # blend = pyramids.pyramid_blending(res_Trump, res_Biden, mask)
    # cv2.imwrite('Biden&Trump.jpg', pyramids.reconstruct(blend))
