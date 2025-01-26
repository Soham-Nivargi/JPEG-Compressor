import cv2
import numpy as np
from utils import helper

class Encoder:
    def __init__(self, image):
        self.image = image.astype(np.int32) - 128
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def get_patches(self):
        pad_height = (8 - (self.height % 8)) % 8
        pad_width = (8 - (self.width % 8)) % 8

        self.padded_image = cv2.copyMakeBorder(
            self.image,
            top=pad_height//2, bottom=pad_height - pad_height//2,
            left=pad_width//2, right=pad_width - pad_width//2,
            borderType=cv2.BORDER_CONSTANT,
            value=0 
        )

        self.patches = self.padded_image.reshape(-1, 8, 8)
        return self.patches

    def get_dcts(self):
        self.dcts = []
        for block in self.patches:
            self.dcts.append(helper.get_dct(block))
        self.dcts = np.array(self.dcts)
        return self.dcts

    def get_quantization(self, quality):
        self.quantization = np.int32(np.round(self.dcts/helper.get_quality_matrix(quality)))
        return self.quantization

    def encode(self, quality):
        self.get_patches()
        self.quality = quality
        self.get_dcts()
        self.get_quantization(quality)
        self.encodered_arr = self.quantization.flatten()
        return self.encodered_arr


class Color_Encoder:
    def __init__(self, file_name):
        self.file_name = file_name
        image_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(file_name, cv2.IMREAD_COLOR)
        image_color_to_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        self.is_gray = True
        self.height, self.width = image_gray.shape

        if np.array_equal(image_gray, image_color_to_gray):
            self.image = image_gray.astype(np.int32)
        else:            
            pad_height = (8 - (self.height % 8)) % 8
            pad_width = (8 - (self.width % 8)) % 8
            padded_image = cv2.copyMakeBorder(
                image_color,
                top=pad_height//2, bottom=pad_height - pad_height//2,
                left=pad_width//2, right=pad_width - pad_width//2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            self.image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2YCrCb).astype(np.int32)
            self.is_gray = False

    def encode(self, quality):
        self.quality = quality
        self.compressed_image = []
        if self.is_gray:
            en_obj = Encoder(self.image)
            self.compressed_image.extend(en_obj.encode(quality))
        else:
            height , width, _ = self.image.shape
            # Y
            en_obj = Encoder(self.image[:, :, 0])
            self.compressed_image.extend(en_obj.encode(quality))
            
            # Cb
            image = self.image[:, :, 1].reshape(height//2, 2, width//2, 2).swapaxes(1, 2).reshape(height//2, width//2, 4)
            image = np.round(image.mean(axis = 2)).astype(np.int32)
            en_obj = Encoder(image)
            self.compressed_image.extend(en_obj.encode(quality))
            
            # Cr
            image = self.image[:, :, 2].reshape(height//2, 2, width//2, 2).swapaxes(1, 2).reshape(height//2, width//2, 4)
            image = np.round(image.mean(axis = 2)).astype(np.int32)
            en_obj = Encoder(image)
            self.compressed_image.extend(en_obj.encode(quality))

    def write(self, file_name):
        helper.huffman_compress(self.compressed_image, [self.is_gray , self.quality, self.height, self.width], file_name)

    def print_info(self):
        pass
