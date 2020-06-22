import numpy as np
import utils
import random
import cv2
import model as modellib
import logging
import os
import time
import sys
import keras.backend as K

from config import Config

import model as modellib
import tensorflow as tf

import skimage.color
import skimage.io
import skimage.transform

import argparse
random.seed(a=1)


def read_image(image_file):
    image = skimage.io.imread(image_file)
    if image.ndim == 3:
        image = skimage.color.rgb2gray(image)
    return image

def resize_image(image, size):
    image = skimage.transform.resize(image,size)
    return image

def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return (images.astype(np.float32) - config.MEAN_PIXEL)

def preprocess(image_file):

    # Load, grayscale and resize image
    image = read_image(image_file)
    image = resize_image(image,(240,320))

    (height, width) = image.shape

    marginal = config.MARGINAL_PIXEL
    patch_size = config.PATCH_SIZE

    # create random point P within appropriate bounds
    y = random.randint(marginal, height - marginal - patch_size)
    x = random.randint(marginal, width - marginal - patch_size)
    # define corners of image patch
    top_left_point = (x, y)
    bottom_left_point = (x, patch_size + y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x + patch_size, y)

    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-marginal, marginal),
                                        point[1] + random.randint(-marginal, marginal)))

    y_grid, x_grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

    # Two branches. The CNN try to learn the H and inv(H) at the same time. So in the first branch, we just compute the
    #  homography H from the original image to a perturbed image. In the second branch, we just compute the inv(H)
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    warped_image = cv2.warpPerspective(image, np.linalg.inv(H), (image.shape[1], image.shape[0]))

    img_patch_ori = image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
    img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1],
                                    top_left_point[0]:bottom_right_point[0]]

    point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float32), H).squeeze()
    diff_branch1 = point_transformed_branch1 - point
    diff_x_branch1 = diff_branch1[:, 0]
    diff_y_branch1 = diff_branch1[:, 1]

    diff_x_branch1 = diff_x_branch1.reshape((image.shape[0], image.shape[1]))
    diff_y_branch1 = diff_y_branch1.reshape((image.shape[0], image.shape[1]))

    pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                            top_left_point[0]:bottom_right_point[0]]

    pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                            top_left_point[0]:bottom_right_point[0]]

    img_patch_ori = mold_image(img_patch_ori, config)
    img_patch_pert = mold_image(img_patch_pert, config)
    image_patch_pair = np.zeros((patch_size, patch_size, 2))
    image_patch_pair[:, :, 0] = img_patch_ori
    image_patch_pair[:, :, 1] = img_patch_pert


    base_four_points = np.asarray([x, y,
                                    x, patch_size + y,
                                    patch_size + x, patch_size + y,
                                    x + patch_size, y])

    perturbed_four_points = np.asarray(perturbed_four_points)
    perturbed_base_four_points = np.asarray([perturbed_four_points[0, 0], perturbed_four_points[0, 1],
                                                perturbed_four_points[1, 0], perturbed_four_points[1, 1],
                                                perturbed_four_points[2, 0], perturbed_four_points[2, 1],
                                                perturbed_four_points[3, 0], perturbed_four_points[3, 1]])
    # Init batch arrays
    batch_image_patch_pair = np.zeros((1,) + (config.PATCH_SIZE, config.PATCH_SIZE, 2),
                                        dtype=np.float32)
    batch_base_four_points = np.zeros((1, 8), dtype=np.float32)

    batch_perturbed_base_four_points = np.zeros((1, 8), dtype=np.float32)

    # Add to batch
    batch_image_patch_pair[0, :, :, :] = image_patch_pair
    batch_base_four_points[0, :] = base_four_points
    batch_perturbed_base_four_points[0, :] = perturbed_base_four_points

    inputs = [batch_image_patch_pair]
    return inputs, batch_base_four_points, batch_perturbed_base_four_points

def metric_paf(Y_pred, PATCH_SIZE, base_four_points, perturbed_base_four_points):

    Y_pred_in_loop = Y_pred[0, :, :, :]
    base_four_points_in_loop = base_four_points[0, :]
    perturbed_base_four_points_in_loop = perturbed_base_four_points[0, :]

    # define corners of image patch
    top_left_point = (base_four_points_in_loop[0], base_four_points_in_loop[1])
    bottom_left_point = (base_four_points_in_loop[2], base_four_points_in_loop[3])
    bottom_right_point = (base_four_points_in_loop[4], base_four_points_in_loop[5])
    top_right_point = (base_four_points_in_loop[6], base_four_points_in_loop[7])

    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

    perturbed_top_left_point = (perturbed_base_four_points_in_loop[0], perturbed_base_four_points_in_loop[1])
    perturbed_bottom_left_point = (perturbed_base_four_points_in_loop[2], perturbed_base_four_points_in_loop[3])
    perturbed_bottom_right_point = (perturbed_base_four_points_in_loop[4], perturbed_base_four_points_in_loop[5])
    perturbed_top_right_point = (perturbed_base_four_points_in_loop[6], perturbed_base_four_points_in_loop[7])

    perturbed_four_points = [perturbed_top_left_point, perturbed_bottom_left_point, perturbed_bottom_right_point, perturbed_top_right_point]

    predicted_pf_x1 = Y_pred_in_loop[:, :, 0]
    predicted_pf_y1 = Y_pred_in_loop[:, :, 1]

    pf_x1_img_coord = predicted_pf_x1
    pf_y1_img_coord = predicted_pf_y1


    y_patch_grid, x_patch_grid = np.mgrid[0:config.PATCH_SIZE, 0:config.PATCH_SIZE]

    patch_coord_x = x_patch_grid + top_left_point[0]
    patch_coord_y = y_patch_grid + top_left_point[1]

    points_branch1 = np.vstack((patch_coord_x.flatten(), patch_coord_y.flatten())).transpose()
    mapped_points_branch1 = points_branch1 + np.vstack(
        (pf_x1_img_coord.flatten(), pf_y1_img_coord.flatten())).transpose()


    original_points = np.vstack((points_branch1))
    mapped_points = np.vstack((mapped_points_branch1))

    H_predicted = cv2.findHomography(np.float32(original_points), np.float32(mapped_points), cv2.RANSAC, 10)[0]

    predicted_delta_four_point = cv2.perspectiveTransform(np.asarray([four_points], dtype=np.float32),
                                                                H_predicted).squeeze() - np.asarray(perturbed_four_points)

    result = np.mean(np.linalg.norm(predicted_delta_four_point, axis=1))
    return result

def inference_PFNet(model, image_file):

    X, base_four_points, perturbed_base_four_points = preprocess(image_file)
    
    t_start = time.time()

    # inference
    result = model.keras_model.predict(X)
    print(result)

    # measure
    mace_ = metric_paf(result,config.PATCH_SIZE,base_four_points,perturbed_base_four_points)
    t_prediction = (time.time() - t_start)

    print("Total time: ", time.time() - t_start)

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PFNet demo on image.')
    parser.add_argument('--model', required=True,metavar="/path/to/weights.h5",help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=True,metavar="/path/to/img.png",help='Image for inference')
    args = parser.parse_args()

    config = Config()
    config.NAME='inference'

    # Create model
    model = modellib.DensePerspective(mode="inference", config=config, model_dir='')

    # Load weights
    model_path = args.model
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Run inference
    print("Running PFNet inference on {}".format(args.image))
    inference_PFNet(model, args.image)

    print("COMPLETED SUCCESSFULLY")
