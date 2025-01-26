from scipy.fftpack import dct, idct
import numpy as np
# import matplotlib.pyplot as plt
import dahuffman
import pickle
import itertools
from bitarray import bitarray
import cv2

def get_bitarray(i):
    return bin(i).split('b')[1]

def twos_complement(bits):
    bits = bits.replace('1', '*')
    bits = bits.replace('0', '1')
    bits = bits.replace('*', '0')
    return bits


def encode_arr(b):
    b = list(b)
    arr = b.copy()
    arr.append(1)
    output = []
    couter = 0
    bit_str = ''
    for i in arr:
        bits = get_bitarray(i)
        if i != 0:
            output.append([couter, len(bits)])
            if i < 0:
                bits = twos_complement(bits)
            bit_str += bits
            couter = 0
        else:
            couter = couter + 1
    return output, bit_str

def decode_arr(run_length_encoding, bits_stream):
    decoded_arr = []
    for i in run_length_encoding:
        n, m = i
        decoded_arr.extend([0]*n)
        current_bits = bits_stream[:m] 
        if current_bits[0] == '0':
            current_bits = twos_complement(current_bits)
            int_value = -int(current_bits, 2)
        else:
            int_value = int(current_bits, 2)

        decoded_arr.append(int_value)
        bits_stream = bits_stream[m:]

    return decoded_arr[:-1]

def huffman_compress(data, info, file_path):
    patches = np.array(data).reshape(-1, 8, 8)
    zig_zac_list = []
    for patch in patches:
        zig_zac_list.extend(get_zig_zag_arr(patch))
    data = np.array(zig_zac_list)

    run_length_encoding, bits_stream = encode_arr(data)
    run_length_encoding = np.array(run_length_encoding).flatten()
    codec = dahuffman.HuffmanCodec.from_data(run_length_encoding)
    compressed_data = codec.encode(run_length_encoding)
    package = [codec, bitarray(bits_stream), compressed_data, info]

    with open(file_path, "wb") as file:
        pickle.dump(package, file)
    
def huffman_decompress(file_path):
    with open(file_path, "rb") as file:
        loaded_package = pickle.load(file)

    loaded_codec = loaded_package[0]
    loaded_bitstream = loaded_package[1].to01()
    loaded_compressed_data = loaded_package[2]
    info = loaded_package[3]

    run_length_encode = loaded_codec.decode(loaded_compressed_data)
    run_length_encode = np.array(run_length_encode).reshape(-1, 2)
    data = decode_arr(run_length_encode, loaded_bitstream)
    encoded_array = list(map(int, data))
    original_array = np.array(encoded_array)
    patches = np.array(original_array).reshape(-1, 64)
    reverse_zig_zac_list = []
    for patch in patches:
        reverse_zig_zac_list.extend(inverse_zig_zag(patch, 8, 8))
    original_array = np.array(reverse_zig_zac_list)
    return original_array.flatten(), info

    return original_array.flatten(), info


def get_dct(arr):
    return cv2.dct(np.array(arr, dtype=np.float32))
    # return dct(dct(arr.T, norm='ortho', type=2).T, norm='ortho', type=2)

def get_idct(arr):
    # return idct(idct(arr.T, norm='ortho', type=2).T, norm='ortho', type=2)
    return cv2.idct(arr)

def show_image(img):
    if len(img.shape) == 3:
        plt.imshow(img[:,:,::-1])
    else:
        plt.imshow(img, cmap='gray')
    plt.show()


def get_quality_matrix(quality):
    quality_50 = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    return quality_50*50/quality

def get_zig_zag_arr(arr2d):
    h, w = arr2d.shape
    arr = []
    for d in range(h + w - 1):
        diagonal = []
        for i in range(max(0, d - w + 1), min(h, d + 1)):
            diagonal.append(arr2d[i, d - i])
        if d%2 == 0:
            diagonal = diagonal[::-1]
        arr.extend(diagonal)
    return arr



def inverse_zig_zag(arr, rows, cols):   
    result = np.zeros((rows, cols), dtype=int)
    index = 0

    for diag in range(rows + cols - 1):
        if diag % 2 == 0:
            r = min(diag, rows - 1)
            c = diag - r
            while r >= 0 and c < cols:
                result[r, c] = arr[index]
                index += 1
                r -= 1
                c += 1
        else:             
            c = min(diag, cols - 1)
            r = diag - c
            while c >= 0 and r < rows:
                result[r, c] = arr[index]
                index += 1
                r += 1
                c -= 1

    return result


def remove_trailing_zeros(arr):
    last_non_zero_index = np.nonzero(arr)[0][-1] if np.any(arr) else -1
    return arr[:last_non_zero_index + 1] if last_non_zero_index != -1 else np.array([])

