from shutil import copyfile
import cv2
from Asbestos_Utils import name_list, load_image_label,get_file_extension
from os.path import exists, join

from os import makedirs


def create_thick_fibers(label, thickness):
    """
    This function takes a labeled ground truth image of fibers and returns an image with the labeled ground
    truth with thicker lines
    :param label: A numpy ndarray with the label ground truth
    :param thickness: The thickness that will be used for the contours in the image
    :return: Returns a numpy ndarray with contours of the fibers with the defined thickness
    """
    # Make a copy of your label to avoid overwriting
    label_copy = label
    # Get the contour for each of the fibers
    ret, thresh = cv2.threshold(label_copy, 127, 255, cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    label_copy = cv2.drawContours(label_copy, contours, -1, (0, 255, 0), thickness=thickness)
    label_copy = cv2.drawContours(label_copy, contours, -1, (0, 255, 0), thickness=-1)
    return label_copy


def save_thick_fibers(read_path, save_path, thickness, write_images=True):
    """
    Takes the images in the read path together with the ground truth labels. Produces and saves
    new images with thicker ground truth label saved in the save path.
    :param read_path: The path were the original ground truths are
    :param save_path: The path were the new thicker ground truths will be saved
    :param thickness: The thickness for the new ground truth images
    :param load_type: The type of the original images saved
    :param write_images: This flag determines if the original images will be also copied into the path (the input data)
    :return: None
    """

    # Get the image names in the read path
    names = name_list(read_path)
    if not (exists(save_path)):
        makedirs(save_path)  # Creates the folder if the save path doesn't exist

    # For each of the images produce a new image with the thicker ground truth
    for name in names:
        print('Converting image %s...\n' % name)
        _, label = load_image_label(read_path, name)
        thick_label = create_thick_fibers(label, thickness)
        cv2.imwrite(join(save_path, name.replace(get_file_extension(name), '_FIB' + get_file_extension(name))), thick_label)
        if write_images:
            copyfile(join(read_path, name), join(save_path, name))
