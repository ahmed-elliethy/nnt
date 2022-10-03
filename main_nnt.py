# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright © 2022 Ahmed Elliethy.
#
# All rights reserved.
#
# This software should be used, reproduced and modified only for informational and nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package)
#

import argparse
import cv2
from skimage import metrics

from NeuralNoiseprintTransfer.core import transfer_noise_all, vis_results
from NeuralNoiseprintTransfer.util import transfer_color_to_gray

parser = argparse.ArgumentParser(description="main_injection")
parser.add_argument("--img_forged_filename", type=str, default='Demo/splicing-01.png', help="Forged image file name")
parser.add_argument("--img_auth_filename", type=str, default='Demo/normal-01.png', help="Authentic image file name")
parser.add_argument("--method_name", type=str, default="injection", help='Method name. Can be injection or optimization')
parser.add_argument("--out_filename", type=str, default='out.png', help="Output image file name")
parser.add_argument("--visualize_results", type=bool, default='True', help="Visualize results")
opt = parser.parse_args()

method_name = opt.method_name
img_forged_filename = opt.img_forged_filename
img_auth_filename = opt.img_auth_filename
out_filename = opt.out_filename
visualize_results = opt.visualize_results

# check for errors
if method_name != 'injection' and method_name != 'optimization':
    method_name = 'injection'

print("Forged: " + img_forged_filename + " ====== " + "Authentic: " + img_auth_filename)

original_forged_img, auth_img, generated_img = transfer_noise_all(img_auth_filename, img_forged_filename, method_name)

generated_img_color = transfer_color_to_gray(img_forged_filename, generated_img)
cv2.imwrite(out_filename, generated_img_color * 255)

forged_img = cv2.imread(img_forged_filename)
generated_img = cv2.imread(out_filename)
psnr = metrics.peak_signal_noise_ratio(forged_img, generated_img)
ssimm = metrics.structural_similarity(forged_img, generated_img, multichannel=True)
print('PSNR =  {}, and SSIM = {}'.format(psnr, ssimm))

if visualize_results:
    vis_results(out_filename, img_forged_filename)
