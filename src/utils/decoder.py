## This file contains Decoder class
from utils import helper
import numpy as np
import cv2

class Decoder:
    def __init__(self, data, quality, height, width):
        self.data = np.array(data)
        self.quality = quality
        self.height = height
        self.width = width

    def decode(self):
        self.quantization = self.data.reshape(-1, 8, 8)
        self.dequant = self.quantization * helper.get_quality_matrix(self.quality)

        self.patches = []
        self.real_image = []
        for block in self.dequant:
            self.patches.append(helper.get_idct(block))

        self.patches = np.array(self.patches) + 128
        self.image = self.patches.flatten().astype(np.int32)

        self.image = np.clip(self.image, 0, 255)


        pad_height = (8 - (self.height % 8)) % 8
        pad_width = (8 - (self.width % 8)) % 8
        
        self.image = self.image.reshape(self.height+pad_height, self.width+pad_width)
        self.image = self.image[pad_height//2:pad_height//2 + self.height, pad_width//2:pad_width//2+self.width]
        return self.image

    def print_info(self):
        print(self.image.shape)



class Color_Decoder:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data, self.info = helper.huffman_decompress(file_name)

    def decode(self):
        if self.info[0]:
            decode_obj = Decoder(self.data, self.info[1], self.info[2], self.info[3])
            return decode_obj.decode()
        else:
            original_height = self.info[2]
            original_width = self.info[3]
            pad_height = (8 - (original_height % 8)) % 8
            pad_width = (8 - (original_width % 8)) % 8
            height = original_height + pad_height
            width = original_width + pad_width
            data_y = self.data[:height*width] 
            
            data = self.data[height*width:]
            decode_obj = Decoder(data_y, self.info[1], self.info[2], self.info[3])
            image_y = decode_obj.decode()

            data_cb = data[:len(data)//2]
            data_cr = data[len(data)//2:]

            decode_obj = Decoder(data_cb, self.info[1], height//2, width//2)
            image_cb = decode_obj.decode()
            h, w = image_cb.shape
            image_cb = image_cb.reshape(h, w, 1).repeat(4, axis = 2)
            image_cb = image_cb.reshape(h, w, 2, 2).swapaxes(1, 2).reshape(h*2, w*2)

            decode_obj = Decoder(data_cr, self.info[1], height//2, width//2)
            image_cr = decode_obj.decode()
            h, w = image_cr.shape
            image_cr = image_cr.reshape(h, w, 1).repeat(4, axis = 2)
            image_cr = image_cr.reshape(h, w, 2, 2).swapaxes(1, 2).reshape(h*2, w*2)

            image = np.zeros((image_y.shape[0], image_y.shape[1], 3))
            image[:, :, 0] = image_y
            image[:, :, 1] = image_cb[pad_height//2:pad_height//2 + original_height, pad_width//2:pad_width//2+original_width]
            image[:, :, 2] = image_cr[pad_height//2:pad_height//2 + original_height, pad_width//2:pad_width//2+original_width]

            self.image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
            return self.image[pad_height//2:pad_height//2+original_height, pad_width//2:pad_width//2+original_width]
    
    def print_info(self):
        pass
    
