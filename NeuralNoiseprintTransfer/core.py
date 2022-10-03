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
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import Normalize
from torch import optim
from torchvision.utils import save_image

from NeuralNoiseprintTransfer.noise_injector_model import NoiseInjectionModel
from NeuralNoiseprintTransfer.noise_print_extract_model import DnCnnNpPt
from NeuralNoiseprintTransfer.util import image_loader
from NoiseprintOriginal.noiseprint.noiseprint_blind import noiseprint_blind_post
from NoiseprintOriginal.noiseprint.utility.utilityRead import imread2f

erodeKernSize = 15
dilateKernSize = 11

slide = 500  # 3072
largeLimit = 250000  # 9437184
overlap = 34

SAVE_MODEL_PATH = "NeuralNoiseprintTransfer/Model/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_print_extract_network = DnCnnNpPt()
transformer_network = NoiseInjectionModel()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    noise_print_extract_network = nn.DataParallel(noise_print_extract_network)
    transformer_network = nn.DataParallel(transformer_network)

noise_print_extract_network.to(device).eval()

final_path = SAVE_MODEL_PATH + "transformer_weight_reverse.pth"
if os.path.exists(final_path):
    transformer_network.load_state_dict(torch.load(final_path, map_location=torch.device(device)))

transformer_network.to(device).eval()


def get_noise_features_at_layer(img, layer='first', method='injection'):
    if method == 'injection':
        with torch.no_grad():  # this can save much memory
            if layer == 'first':
                return noise_print_extract_network.get_output_at_first(img)
            else:
                return noise_print_extract_network.get_output_at_end(img)
    else:
        if layer == 'first':
            return noise_print_extract_network.get_output_at_first(img)
        else:
            return noise_print_extract_network.get_output_at_end(img)


def get_noise(img):
    with torch.no_grad():  # this can save much memory
        return noise_print_extract_network(img)


def transfer_noise_all(source_noise_f_name, original_target_f_name, method='injection'):
    source_noise_img, _ = imread2f(source_noise_f_name, channel=1)
    original_target_img, _ = imread2f(original_target_f_name, channel=1)
    if source_noise_img.shape != original_target_img.shape:
        return None, None, None

    if source_noise_img.shape[0] * source_noise_img.shape[1] > largeLimit:
        # for large image the network is executed windows with partial overlapping
        res = np.zeros((source_noise_img.shape[0], source_noise_img.shape[1]), np.float32)
        for index0 in range(0, source_noise_img.shape[0], slide):
            index0start = index0 - overlap
            index0end = index0 + slide + overlap

            for index1 in range(0, source_noise_img.shape[1], slide):
                index1start = index1 - overlap
                index1end = index1 + slide + overlap
                source_noise_img_clip = source_noise_img[max(index0start, 0): min(index0end, source_noise_img.shape[0]), \
                                        max(index1start, 0): min(index1end, source_noise_img.shape[1])]
                original_target_img_clip = original_target_img[
                                           max(index0start, 0): min(index0end, original_target_img.shape[0]), \
                                           max(index1start, 0): min(index1end, original_target_img.shape[1])]

                res_clip = transfer_noise_injection(source_noise_img_clip, original_target_img_clip, method)
                res_clip = res_clip.cpu().detach().numpy().T
                res_clip = res_clip[:, :, :, 0].squeeze().T

                if index0 > 0:
                    res_clip = res_clip[overlap:, :]
                if index1 > 0:
                    res_clip = res_clip[:, overlap:]
                res_clip = res_clip[:min(slide, res_clip.shape[0]), :min(slide, res_clip.shape[1])]

                res[index0: min(index0 + slide, res.shape[0]), \
                index1: min(index1 + slide, res.shape[1])] = res_clip

                cv2.imwrite("res_part_image.png", res * 255)
    else:
        res = transfer_noise_injection(source_noise_img, original_target_img, method)
        res = res.cpu().detach().numpy().T
        res = res[:, :, :, 0].squeeze().T
    return original_target_img, source_noise_img, res


def transfer_noise_injection(auth_img, forged_img, method):
    auth_img = image_loader(auth_img, device)
    forged_img = image_loader(forged_img, device)

    if method == 'injection':
        forged_features = get_noise_features_at_layer(forged_img)
        auth_features = get_noise_features_at_layer(auth_img, layer='end')
        generated_image = transformer_network(forged_features, 0.01 * auth_features)
        return generated_image

    elif method == 'optimization':
        print("============================================================")
        pixel_weight = 10
        content_weight = 10
        noise_weight = 1
        adam_lr = 0.006
        optimization_epochs = 500

        zero_image = forged_img.clone()
        generated_image = zero_image.requires_grad_(True)

        # using adam optimizer, and it will update the generated image not the model parameter
        optimizer = optim.Adam([generated_image], lr=adam_lr)
        mse_loss = nn.MSELoss().to(device)

        forged_features = get_noise_features_at_layer(forged_img, method=method)
        auth_features = get_noise_features_at_layer(auth_img, layer='end', method=method)
        # iterating
        for e in range(optimization_epochs):
            generated_features_content = get_noise_features_at_layer(generated_image, method=method)
            generated_features_noise = get_noise_features_at_layer(generated_image, layer='end', method=method)

            # Image Loss
            pixel_loss = pixel_weight * mse_loss(forged_img, generated_image)
            pixel_loss_val = pixel_loss.cpu().detach()

            # Content Loss
            content_loss = content_weight * mse_loss(forged_features, generated_features_content)
            content_loss_val = content_loss.cpu().detach()

            # Noise Loss
            noise_loss = noise_weight * mse_loss(auth_features, generated_features_noise)
            noise_loss_val = noise_loss.cpu().detach()

            total_loss = pixel_loss + content_loss + noise_loss
            total_loss_val = pixel_loss_val + content_loss_val + noise_loss_val
            # optimize the pixel values of the generated image and back-propagate the loss
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # print the image and save it after each 100 epoch
            if not (e % 10):
                print("pixel_loss = {}, content_loss = {}, noise_loss = {}, total_loss = {}".format(pixel_loss_val,
                                                                                                    content_loss_val,
                                                                                                    noise_loss_val,
                                                                                                    total_loss_val))
                save_image(generated_image, "gen_final.png")

        return generated_image


def vis_results(generated_image_filename, forged_image_filename):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    generated_img, _ = imread2f(generated_image_filename, channel=1)
    gen_noise = get_noise(image_loader(generated_img, device))
    gen_noise_img = gen_noise.cpu().detach().numpy().T
    gen_noise_img = gen_noise_img[:, :, :, 0].squeeze().T
    _, mapp_gen, valid, range0, range1, imgsize, other = noiseprint_blind_post(gen_noise_img, generated_img)

    forged_img, _ = imread2f(forged_image_filename, channel=1)
    org_noise = get_noise(image_loader(forged_img, device))
    org_noise_img = org_noise.cpu().detach().numpy().T
    org_noise_img = org_noise_img[:, :, :, 0].squeeze().T
    _, mapp_org, valid, range0, range1, imgsize, other = noiseprint_blind_post(org_noise_img, forged_img)

    plt.subplot(2, 3, 1)
    plt.title('Original forged image')
    plt.imshow(forged_img, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.imshow(org_noise_img, cmap='gray', norm=Normalize(vmin=0.0, vmax=1.0))
    plt.title('Forged noiseprint')

    plt.subplot(2, 3, 3)
    plt.imshow(mapp_org, clim=[np.nanmin(mapp_org), np.nanmax(mapp_org)], cmap='jet')
    plt.title('Forged noiseprint heatmap')

    plt.subplot(2, 3, 4)
    plt.title('Generated image')
    plt.imshow(generated_img, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.imshow(gen_noise_img, cmap='gray', norm=Normalize(vmin=0.0, vmax=1.0))
    plt.title('Generated noiseprint')

    plt.subplot(2, 3, 6)
    plt.imshow(mapp_gen, clim=[np.nanmin(mapp_org), np.nanmax(mapp_org)], cmap='jet')
    plt.title('Generated noiseprint heatmap')

    plt.savefig(generated_image_filename + '_results.eps', bbox_inches='tight', pad_inches=0)

    plt.show()
