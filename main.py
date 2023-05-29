import cv2
import numpy as np

import pyramids
from pyramids import Pyramids
import scipy.io as scio


def save_image(img, name):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    cv2.imwrite(name, (img * 255).astype(np.uint8))
    return img


def DFT(img):
    return np.fft.fftshift(np.fft.fft2(img))


def IDFT(img):
    return np.fft.ifft2(np.fft.ifftshift(img))


def visualize_DFT(img, name=None):
    img_fft = DFT(img)
    img_fft = np.log(np.abs(img_fft))
    img_fft = save_image(img_fft, name)
    return img_fft


def get_inner_radius(img):
    kernel_size = 5
    r = []
    ori_size = img.shape[0]
    while img.shape[0] > 64:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1)
        edge = cv2.Canny((img * 255).astype(np.uint8), 40, 90)
        max_r = 0
        for x in range(edge.shape[0]):
            for y in range(edge.shape[1]):
                if edge[x][y] == 255:
                    distance = np.sqrt((x - edge.shape[0] // 2) ** 2 + (y - edge.shape[1] // 2) ** 2)
                    if max_r < distance:
                        max_r = distance
        r.append(max_r * ori_size / img.shape[0])
        save_image(edge, f'PSF_edge_{img.shape[0]}.jpg')
        img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    r = np.mean(r) + kernel_size / 2
    return (3.83 * ori_size) / (2 * np.pi * r)


def HoughTransform(img, delta_theta=1.0, delta_rho=1.0):
    theta = np.arange(-90, 90, delta_theta)
    D = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    rho = np.arange(-D, D, delta_rho)
    cosine = np.cos(theta / 180 * np.pi)
    sine = np.sin(theta / 180 * np.pi)
    accumulator = np.zeros((len(rho), len(theta)))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] == 255:
                r = x * cosine + y * sine
                r = ((r + D) / (2 * D) * len(rho - 1)).astype(np.int32)
                for i in range(len(theta)):
                    accumulator[r[i], i] += 1
    return accumulator, theta, rho


def get_motion_blur_params(img):
    kernel_size = 5
    ori_size = img.shape[0]
    distance = []
    angle = []
    while img.shape[0] > 64:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0.7)
        edge = cv2.Canny((img * 255).astype(np.uint8), 72, 130)
        accumulator, theta, rho = HoughTransform(edge, delta_theta=1, delta_rho=1)
        save_image(accumulator, 'MB_accumulator.jpg')
        _, c = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        r = np.argsort(accumulator[:, c])[-2:]
        rho_1, rho_2 = rho[r]
        angle.append(theta[c])
        distance.append((np.abs(rho_1 - rho_2) + kernel_size / 2) * ori_size / img.shape[0])
        save_image(edge, f'MB_edge_{img.shape[0]}.jpg')
        img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    theta = np.mean(angle)
    distance = np.mean(distance)
    b = -1 / (distance / 2 / np.sin(theta / 180 * np.pi))
    a = -1 / (distance / 2 / np.cos(theta / 180 * np.pi))
    # a = -0.021
    # b = 97 / 57 * 0.021
    # a = -2 / 15
    # b = 2 * np.sqrt(3) / 15
    T = 250
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
    save_image(np.abs(kernel), 'MB_kernel.jpg')
    return kernel


def restore_Trump(img, R=None, K=50):
    """
    :param img: The image of Trump with 3 channels
    :param R: manually measured radius of the inner circle of the PSF kernel
    :param K: The inverse of SNR i.e. noise-to-signal-ratio
    :return: The restored image of Trump
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = visualize_DFT(img_gray, name="Trump_fft.jpg")
    R = get_inner_radius(img_fft) if R is None else R
    PSF_kernel = get_PSF_kernel(img_fft.shape[0], R)
    restored_img = Wiener_filter(img, PSF_kernel, K)
    restored_img = np.fft.ifftshift(restored_img, axes=(0, 1))
    return restored_img


def restore_Biden(img, a=None, b=None, T=None, K=50):
    """
    :param img: image of Biden with 3 channels
    :param a: proportional to extent of motion blur in vertical direction
    :param b: proportional to extent of motion blur in horizontal direction
    :param T: exposure time
    :param K: inverse of SNR i.e. noise-to-signal-ratio
    :return: restored image of Biden
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fft = visualize_DFT(img_gray, name="Biden_fft.jpg")
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
    save_image(res_Trump, 'restored_Trump.jpg')
    save_image(res_Biden, 'restored_Biden.jpg')
    pyramids = Pyramids()
    blend = pyramids.pyramid_blending(res_Trump, res_Biden, mask)
    cv2.imwrite('Biden&Trump.jpg', pyramids.reconstruct(blend))
