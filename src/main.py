import argparse
import cv2
from utils import helper, encoder, decoder

parser = argparse.ArgumentParser("This is an image compressor program")
parser.add_argument("--encode", "-e", action='store_true')
parser.add_argument("--input-file", "-i", type=str, default="")
parser.add_argument("--encoded-data-file", "-r", type=str, default="")
parser.add_argument("--quality", "-q", type=int, default=50)
parser.add_argument("--decode", "-d", action='store_true')
parser.add_argument("--output-file", "-o", type=str, default="")

args = parser.parse_args()


if args.encode and args.input_file and args.encoded_data_file:
    file_name = args.input_file
    encoder_obj = encoder.Color_Encoder(file_name)
    encoder_obj.encode(args.quality)
    encoder_obj.write(args.encoded_data_file)
    encoder_obj.print_info()

if args.decode and args.encoded_data_file:
    file_name = args.encoded_data_file
    decoder_obj = decoder.Color_Decoder(file_name)
    img = decoder_obj.decode()
    decoder_obj.print_info()
    if args.output_file:
        cv2.imwrite(args.output_file, img)
