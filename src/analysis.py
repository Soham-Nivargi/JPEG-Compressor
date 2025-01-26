# This file contains the code for the analysis of the data
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = '../results'
dataset = '3d-2d'
#dataset = 'coil-20-unproc'
#dataset = 'landscape_images'
#inner_dir = 'gray/'
inner_dir = ''
image_data_path = f'../data/{dataset}/{inner_dir}'
PIL_image_output_path = f'../data/{dataset}-compressed/{inner_dir}PIL/'
OUR_image_output_path = f'../data/{dataset}-compressed/{inner_dir}OUR/'

images_list = os.listdir(image_data_path)
quality_list = [25, 50, 75, 100]

actual_image_size_list = np.array([os.path.getsize(image_data_path + image) for image in images_list])


def iterative_degredation_analysis_our(n, image_path, quality, orig_image_path):
    if(n == 0):
        img1 = Image.open(image_path)
        img2 = Image.open(orig_image_path)
        return np.sqrt(np.mean((np.array(img1) - np.array(img2))**2))
    
    os.system(f'python3 main.py -e -i {image_path} -r temp_dir/out_{n}_{quality} -q {quality} -d -o temp_dir/out_{n}_{quality}.png')

    return iterative_degredation_analysis_our(n-1, f'temp_dir/out_{n}_{quality}.png', quality, orig_image_path)


def iterative_degredation_analysis_pil(n, image_path, quality, orig_image_path):
    if(n == 0):
        img1 = Image.open(image_path)
        img2 = Image.open(orig_image_path)
        return np.sqrt(np.mean((np.array(img1) - np.array(img2))**2))

    img = Image.open(image_path)
    img.save(f'temp_dir/out_{n}_{quality}.jpg', 'JPEG', quality=quality)

    return iterative_degredation_analysis_pil(n-1, f'temp_dir/out_{n}_{quality}.jpg', quality, orig_image_path)

quality_PIL_RMSE_list = []
quality_OUR_RMSE_list = []
quality_PIL_compression_ratio_list = []
quality_OUR_compression_ratio_list = []
quality_PIL_bpp_list = []
quality_OUR_bpp_list = []

for q in quality_list:
    for image in images_list:
        os.system(f'python3 main.py -e -i {image_data_path + image} -r {OUR_image_output_path + image[:-4]}_out_{q} -q {q} -d -o {OUR_image_output_path + image[:-4]}_out_{q}.png')

    image_dimensions_list = []
    for image in images_list:
        img = Image.open(image_data_path + image)
        image_dimensions_list.append(img.size)
        img.save(PIL_image_output_path + image[:-4] + f'_out_{q}.jpg', 'JPEG', quality=q)

    # Compression Ratio
    PIL_compressed_image_size_list = np.array([os.path.getsize(PIL_image_output_path + image[:-4] + f"_out_{q}.jpg") for image in images_list])
    compression_ratio_PIL = actual_image_size_list / PIL_compressed_image_size_list


    OUR_compressed_image_size_list = np.array([os.path.getsize(OUR_image_output_path + image[:-4] + f"_out_{q}") for image in images_list])
    compression_ratio_our = actual_image_size_list / OUR_compressed_image_size_list

    # BPP - Pits Per Pixel
    image_num_pixels_list = np.array([np.prod(np.array(curr_dims)) for curr_dims in image_dimensions_list])
    bpp_PIL = PIL_compressed_image_size_list * 8 / image_num_pixels_list
    bpp_our = OUR_compressed_image_size_list * 8 / image_num_pixels_list

    # RMSE - Root Mean Square Error
    PIL_RMSE_list = []
    OUR_RMSE_list = []
    for image in images_list:
        img = Image.open(image_data_path + image)
        our_img = Image.open(OUR_image_output_path + image[:-4] + f"_out_{q}.png")
        pil_img = Image.open(PIL_image_output_path + image[:-4] + f"_out_{q}.jpg")

        PIL_RMSE_list.append(np.sqrt(np.mean((np.array(img) - np.array(pil_img))**2)))
        OUR_RMSE_list.append(np.sqrt(np.mean((np.array(img) - np.array(our_img))**2)))
    
    quality_PIL_RMSE_list.append(PIL_RMSE_list)
    quality_OUR_RMSE_list.append(OUR_RMSE_list)
    quality_PIL_compression_ratio_list.append(compression_ratio_PIL)
    quality_OUR_compression_ratio_list.append(compression_ratio_our) 
    quality_PIL_bpp_list.append(bpp_PIL)
    quality_OUR_bpp_list.append(bpp_our)
    print(f"Quality: {q}")

quality_PIL_RMSE_list = np.array(quality_PIL_RMSE_list)
quality_OUR_RMSE_list = np.array(quality_OUR_RMSE_list)
quality_PIL_compression_ratio_list = np.array(quality_PIL_compression_ratio_list)
quality_OUR_compression_ratio_list = np.array(quality_OUR_compression_ratio_list)
quality_PIL_bpp_list = np.array(quality_PIL_bpp_list)
quality_OUR_bpp_list = np.array(quality_OUR_bpp_list)

# Mean RMSE VS Quality
plt.scatter(quality_list, np.mean(quality_PIL_RMSE_list, axis=1), color='black')
plt.plot(quality_list, np.mean(quality_PIL_RMSE_list, axis=1), label='PIL')
plt.scatter(quality_list,np.mean(quality_OUR_RMSE_list, axis=1), color='black')
plt.plot(quality_list,np.mean(quality_OUR_RMSE_list, axis=1), label='OUR')
plt.xlabel('Quality')
plt.ylabel('Mean RMSE')
plt.title('RMSE vs Quality')
plt.legend()
plt.savefig(f'{RESULTS_DIR}/{dataset}_rmse_vs_quality.png')
plt.close()

# Mean Compression Ration VS Quality
plt.scatter(quality_list,np.mean(quality_PIL_compression_ratio_list, axis=1), color='black')
plt.plot(quality_list, np.mean(quality_PIL_compression_ratio_list, axis=1), label='PIL')
plt.scatter(quality_list,np.mean(quality_OUR_compression_ratio_list, axis=1), color='black')
plt.plot(quality_list,np.mean(quality_OUR_compression_ratio_list, axis=1), label='OUR')
plt.xlabel('Quality')
plt.ylabel('Mean Compression Ratio')
plt.title('Compression Ratio vs Quality')
plt.legend()
plt.savefig(f'{RESULTS_DIR}/{dataset}_compression_ratio_vs_quality.png')
plt.close()

# BPP vs RMSE for various quality factor for each image
image_PIL_quality_RMSE_list = quality_PIL_RMSE_list.T
image_OUR_quality_RMSE_list = quality_OUR_RMSE_list.T
image_PIL_quality_bpp_list = quality_PIL_bpp_list.T
image_OUR_quality_bpp_list = quality_OUR_bpp_list.T
for i, image in enumerate(images_list):
    plt.scatter(image_PIL_quality_bpp_list[i], image_PIL_quality_RMSE_list[i], color='black')
    plt.plot(image_PIL_quality_bpp_list[i], image_PIL_quality_RMSE_list[i], label='PIL')
    plt.scatter(image_OUR_quality_bpp_list[i], image_OUR_quality_RMSE_list[i], color='black')
    plt.plot(image_OUR_quality_bpp_list[i], image_OUR_quality_RMSE_list[i], label='OUR')
    plt.xlabel('BPP')
    plt.ylabel('RMSE')
    plt.title(f'BPP vs RMSE for {image} with dataset {dataset}')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{dataset}_bpp_vs_rmse_{image[:-4]}.png')
    plt.close()

 

# iterative degredation analysis
image = images_list[0]

num_iteration_list = [2**i for i in range(1, 6)]
for q in quality_list:
    os.system(f'mkdir -p temp_dir')
    RMSE_list_our = []
    RMSE_list_pil = []
    for i in num_iteration_list:
        RMSE_list_our.append(iterative_degredation_analysis_our(i, image_data_path + image, q, image_data_path + image))
        RMSE_list_pil.append(iterative_degredation_analysis_pil(i, image_data_path + image, q, image_data_path + image))
    
    plt.plot(num_iteration_list, RMSE_list_our, label='OUR')
    plt.plot(num_iteration_list, RMSE_list_pil, label='PIL')
    plt.xlabel('Num Iterations')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs Iterations for Quality {q}')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{dataset}_rmse_vs_iterations_{q}.png')
    plt.close()
    os.system(f'rm -rf temp_dir')

    print(f"Quality: {q} done")
 
