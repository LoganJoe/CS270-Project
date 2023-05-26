import numpy as np
import cv2
import scipy.io as scio
from scipy import ndimage


class Pyramids:
    def __init__(self):
        self.kernel = self.gaussian_kernel(9, 4)

    def gaussian_kernel(self, size=5, sigma=1):
        kernel = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        return kernel

    def convolve(self, image, kernel):
        w, h = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2
        image_padded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2, image.shape[2]),
                                dtype=np.float32)
        image_padded[padding:-padding, padding:-padding, :] = image

        for i in range(w):
            for j in range(h):
                result[i, j, :] = np.sum(kernel[..., None] * image_padded[i:i + kernel_size, j:j + kernel_size, :],
                                         axis=(0, 1))

        return image

    def interpolate(self, image):
        image_upscale = np.zeros((image.shape[0] * 2, image.shape[1] * 2, image.shape[2]), dtype=np.float32)
        image_upscale[::2, ::2, :] = image

        return cv2.filter2D(image_upscale, -1, 4 * self.kernel, borderType=cv2.BORDER_CONSTANT)
        # return self.convolve(image_upscale, 4 * self.kernel)

    def decimate(self, image):
        image_downscale = self.convolve(image.astype(np.float32), self.kernel)
        return image_downscale[::2, ::2, :]

    def build_pyramid(self, image):
        G, L = [], []
        G.append(image)

        while (G[-1].shape[0] > 1 and G[-1].shape[1] > 1):
            G.append(self.decimate(G[-1]))

        for i in range(len(G) - 1):
            L.append(G[i] - self.interpolate(G[i + 1]))

        return G[:-1], L

    def pyramid_blending(self, image1, image2, mask):
        G1, L1 = self.build_pyramid(image1)
        G2, L2 = self.build_pyramid(image2)
        GM, LM = self.build_pyramid(mask)

        blend = []

        for gm, la, lb in zip(GM, L1, L2):
            blend.append(gm * la + (1 - gm) * lb)

        return blend

    def reconstruct(self, pyramid):
        rev_pyramid = pyramid[::-1]
        stack = rev_pyramid[0]
        for i in range(1, len(rev_pyramid)):
            stack = self.interpolate(stack) + rev_pyramid[i]
        return stack


if __name__ == '__main__':
    apple = cv2.imread('images/apple.jpg')
    orange = cv2.imread('images/orange.jpg')
    apple = apple[apple.shape[0] // 2 - 256:apple.shape[0] // 2 + 256, apple.shape[1] // 2 - 256:apple.shape[1] // 2 + 256]
    orange = orange[orange.shape[0] // 2 - 256:orange.shape[0] // 2 + 256, orange.shape[1] // 2 - 256:orange.shape[1] // 2 + 256]
    mask = cv2.imread('images/mask.jpg').astype(np.float32) / 255
    pyramids = Pyramids()
    blend = pyramids.pyramid_blending(orange, apple, mask)
    cv2.imwrite('blend.jpg', pyramids.reconstruct(blend))