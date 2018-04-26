from os import listdir, makedirs, remove
from os.path import isdir, join, exists, isfile
import cv2
import numpy as np


# Utilities used for other functions and classes in the UNET network training and evaluation

def get_file_extension(name):
    """
    Obtain the file extension from a file name
    :param name: The string with a file name that includes the extension
    :return: Returns a string with the file extension. Example: .tiff, .jpg, etc.
    """
    extension = '.' + name.split('.')[-1]
    return extension


def name_list(path, extension='.jpg'):
    """
    This function read the names of the interesting files and returns them as a list
    :param path: The path from which the files are read
    :param extension: The extension of the filenames to be read
    :return: Returns a list of filenames with the predetermined extension
    """
    # Get the images names
    image_names = []
    for file_name in listdir(path):
        if not (isdir(join(path, file_name))) and not ('_FIB' in file_name) and (file_name.endswith(extension)):
            image_names.append(file_name)
    return image_names


def load_image(img_path, img_name):
    """
    Loads an image from certain path with certain name
    :param img_path: Image path
    :param img_name: Image name
    :return: Returns an np.ndarray with grayscale pixel values
    """
    # read as grayscale image (type uint8)
    img = cv2.imread(join(img_path, img_name), 0)  # read as gray scale (type: uint8)
    img = img.astype(np.float64)  # otherwise skimage gaussian is strange [auto rescales].
    assert isinstance(img, np.ndarray)
    return img


def save_image(img_path, img_name, image):
    """
    Saves image in the image path
    :param img_path: Image path
    :param img_name: Image name
    :param image: Image to be saved
    :return: None
    """
    # read as grayscale image (type uint8)
    cv2.imwrite(join(img_path, img_name), image)  # read as gray scale (type: uint8)


def load_image_label(path, image_name):
    """
    Loads both the image and the ground truth from certain path
    :param path: Path from which the images are extracted
    :param image_name: Name of the image that is extracted
    :return:
    """
    image = load_image(path, image_name)
    if isfile(join(path, image_name.replace(get_file_extension(image_name), '_FIB' + get_file_extension(
            image_name)))):  # If the file actually has an annotation, load it
        label = load_image(path,
                           image_name.replace(get_file_extension(image_name), '_FIB' + get_file_extension(image_name)))
        label = label.astype(np.uint8)
        __, label = cv2.threshold(label, 127, 255, 0)
    else:  # If the file doesn't have the annotation it is probably because it is an empty image.
        label = np.ones(image.shape) * 255
    return image, label


def save_image_label(path, image_name, image, label):
    """
    Saves the image and label into certain path with certain name
    :param path: Path from which the images are extracted
    :param image_name: Name of the image that is extracted
    :return:none
    """
    save_image(path, image_name, image)
    save_image(path, image_name.replace(get_file_extension(image_name), '_FIB' + get_file_extension(image_name)), label)


def load_folder_images(path, extension='.jpg'):
    """This function loads all the images with certain file_type in the specified path. If the annotations is true then
    it will also load the annotated images with the correct annotation
    Output:
    -image_name: gives a list with the names of all the loaded images.
    -images : gives an array with all the images stored in it.
    -classification : gives an array with all the annotations stored in it.
    """

    image_name = []
    for file_name in listdir(path):
        if not (isdir(join(path, file_name))) and not ('_FIB' in file_name) and (file_name.endswith(extension)):
            image_name.append(file_name)

    images = []  # Initialize the list in which the images are going to be stored
    labels = []  # Initialize the list in which the annotations are going to be stored
    for num_name, name in enumerate(image_name):
        images.append(load_image(path, name))  # Load the image from the specified path with the given name

        # Load the annotations for each fiber image
        if isfile(join(path, name.replace(extension, '_FIB' + extension))):
            label = load_image(path, name.replace(extension, '_FIB' + extension))
            label = label.astype(np.uint8)
            ret, thresh = cv2.threshold(label, 127, 255, 0)
            labels.append(thresh)
        else:  # If the file doesn't have the annotation it is probably because it is an empty image.
            labels.append(np.ones(images[-1].shape) * 255)

    return [image_name, np.array(images), np.array(labels)]


####### IMAGE CROPPING #######


def crop_image(img, size):
    """
    Takes an np.ndarray corresponding to an image and returns a list of smaller regular cropped images with certain size
    :param img: Image to be cropped
    :param size: Size of the output cropped images in the list
    :return: List of np.ndarray with shape (size,size)
    """
    crop_img = []
    # Number of crops along each axis (2 axis x and y)
    crop_num = int(img.shape[0] / size)
    # Append the cropped images to list
    for i in range(crop_num):
        for j in range(crop_num):
            crop_img.append(img[size * i:size * (i + 1), size * j:size * (j + 1)])
    return crop_img


def random_crop_image(img, size):
    """
    Get a random cropped image with certain size from input image
    :param img: Input image
    :param size: Size of output cropped image
    :return: Returns an np.ndarray of shape (size,size)
    """
    x0 = np.random.randint(0, img.shape[0] - size - 1)
    y0 = np.random.randint(0, img.shape[0] - size - 1)
    return img[x0:x0 + size, y0:y0 + size]


def crop_image_save(load_path, save_path, size, load_extension, save_extension='.tiff'):
    """
    This function extracts all the images from the path chosen and crops them to the given size,
    then saves them in the given folder    
    :type load_extension: Extension for the loaded images
    :param load_path: Path from where the images are extracted
    :param save_path: Path where the images are save
    :param size: Size of the cropped output images
    :param load_extension: Extension for the loaded images
    :param save_extension: Extension for the cropped images
    :return: None
    """
    # First the images are loaded from the specified load_path
    image_name, images, labels = load_folder_images(load_path, extension=load_extension)
    if not exists(save_path):
        makedirs(save_path)
    for i in range(len(images)):
        # Each of the images and annotation is then cropped into the specified size
        cropped_image = crop_image(images[i], size)
        cropped_classification = crop_image(labels[i], size)
        for j in range(len(cropped_image)):
            # The images are saved into new files
            cv2.imwrite(join(save_path,
                             image_name[i].replace(load_extension, '_') + str(size) + '_c_' + str(j) + save_extension),
                        cropped_image[j])
            cv2.imwrite(join(save_path, image_name[i].replace(load_extension, '_') + str(size) + '_c_' + str(
                j) + '_FIB' + save_extension), cropped_classification[j])


def remove_non_relevant(path):
    """
    This might be a dangerous function. Take care since it removes file from the path direction.
    It takes all the annotation files and identifies if they actually contain any fiber. If they don't, then the original
    file with the annotation are removed.
    :param path: Path from which the non relevant images will be eliminated
    :return: None
    """
    label_name = []
    for file_name in listdir(path):
        if ('_FIB' in file_name) and not (isdir(join(path, file_name))):
            label_name.append(file_name)
    for i in range(len(label_name)):
        image_name = label_name[i].replace('_FIB', '')  # Get the corresponding names from the original images
        label = load_image(path, label_name[i])  # Load each of the annotations
        if np.sum(label > 0) == np.product(label.shape):  # Check whether the annotation is relevant.
            remove(join(path, image_name))  # Remove the original image
            remove(join(path, label_name[i]))  # Remove the annotation


######## IMAGE AUGMENTATION #######

def image_flipping(img, method):
    """
    Carries out a defined flipping
    0: In both directions x and y
    1: In direction x
    2: In direction y
    Any other number doesn't do anything to the image
    """
    if method == 0:
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
    elif method == 1:
        img = cv2.flip(img, 0)
    elif method == 2:
        img = cv2.flip(img, 1)
    return img


def generate_complete_flipping(load_path, save_path, data_type='.jpg', x_flip=True, y_flip=True,
                               xy_flip=True, percentage_data_generation=1.0):
    """
    This function takes the images in the load path and creates flipped images in the x or y directions if true or
    in both x and y.
    :param load_path:
    :param save_path:
    :param data_type:
    :param x_flip: Flip in x
    :param y_flip: Flip in y
    :param xy_flip: Flip in x AND y
    :param percentage_data_generation: How much of the loaded data is augmented
    :return:
    """
    names_images = name_list(load_path, extension=data_type)  # Get the image names
    if percentage_data_generation != 1.0:
        np.random.shuffle(names_images)  # Take a random permutation of the image names

    # Identify which flipping methods are going to be used
    # 0: xy direction
    # 1: x direction
    # 2: y_direction
    methods = np.array([0, 1, 2])[np.array([xy_flip, x_flip, y_flip])]

    for i in range(int(len(names_images) * percentage_data_generation)):
        # Read the images and annotations to flip them
        image = load_image(load_path, names_images[i])
        if isfile(join(load_path, names_images[i].replace(data_type, '_FIB' + data_type))):
            annotation = load_image(load_path, names_images[i].replace(data_type, '_FIB' + data_type))
        else:
            annotation = np.ones(image.shape) * 255

        # Flip the images and save them
        for method in methods:
            cv2.imwrite(join(save_path, names_images[i].replace(data_type, '_XDA' + str(method) + data_type)),
                        image_flipping(image, method))
            cv2.imwrite(join(save_path, names_images[i].replace(data_type, '_XDA' + str(method) + '_FIB' + data_type)),
                        image_flipping(annotation, method))


def crop_save_remove_from_list(load_path, save_path, list_names, size, load_extension='.jpg', save_extension='.tiff',
                               remove=True):
    """
    Takes the images in the load_path, cuts them to the appropriate size and saves them. If the remove feature is True,
    then it doesn't save the images without fibers.
    :param load_path: Path from which the images are loaded
    :param save_path: Path where the cropped images are saved
    :param list_names: List of the images to be processed
    :param size: Size of the cropped images
    :param load_extension: Loaded images extension
    :param save_extension: Saved images extension
    :param remove: Whether the non-fiber cropped images should be removed
    :return: Total saved images
    """
    total_saved_images = 0
    image_name = list_names
    if not exists(save_path):
        makedirs(save_path)
    for i in range(len(image_name)):
        image = load_image(load_path, image_name[i])
        cropped_image = crop_image(image, size)
        if isfile(join(load_path, image_name[i].replace(load_extension, '_FIB' + load_extension))):
            label = load_image(load_path, image_name[i].replace(load_extension, '_FIB' + load_extension))
            label = label.astype(np.uint8)

            _, label = cv2.threshold(label, 127, 255, 0)
            cropped_label = crop_image(label, size)
        else:
            label = np.ones(image.shape) * 255
            cropped_label = crop_image(label, size)

        for j in range(len(cropped_image)):
            # The images are saved into new files
            if not remove:
                total_saved_images += 1
                cv2.imwrite(join(save_path, image_name[i].replace(load_extension, '_') + str(size) + '_c_' + str(
                    j) + save_extension), cropped_image[j])
                cv2.imwrite(join(save_path, image_name[i].replace(load_extension, '_') + str(size) + '_c_' + str(
                    j) + '_FIB' + save_extension), cropped_label[j])
            elif np.sum(cropped_label[j] > 0) != np.product(
                    cropped_label[j].shape):  # Check whether the annotation is relevant.
                total_saved_images += 1
                cv2.imwrite(join(save_path, image_name[i].replace(load_extension, '_') + str(size) + '_c_' + str(
                    j) + save_extension), cropped_image[j])
                cv2.imwrite(join(save_path, image_name[i].replace(load_extension, '_') + str(size) + '_c_' + str(
                    j) + '_FIB' + save_extension), cropped_label[j])
    return total_saved_images


####### TEXT FILES #######

def save_names(names, path_save, file_output_name):
    names_file = open(join(path_save, file_output_name + '.txt'), 'w+')
    for name in names:
        names_file.write(name + '\n')
    names_file.close()


def load_names(path_load, file_name):
    names_file = open(join(path_load, file_name + '.txt'), 'r+')
    names = names_file.readlines()
    names_file.close()
    for number, name in enumerate(names):
        names[number] = name.replace('\n', '')
    return names

######## PREDICTIONS ########

def prediction_masks(prediction, label):
    TP_mask = np.logical_not((prediction == 0) * (label == 0)) * 1
    FP_mask = np.logical_not((prediction == 0) * (label != 0)) * 1
    FN_mask = np.logical_not((prediction != 0) * (label == 0)) * 1
    colored_mask = np.ones((label.shape[0], label.shape[1], 3))
    colored_mask[:, :, 0] = TP_mask
    colored_mask[:, :, 1] = FP_mask
    colored_mask[:, :, 2] = FN_mask
    return TP_mask, FP_mask, FN_mask, colored_mask

def create_predictions_test(model,path_data,path_prediction,names, threshold, model_name='Generic'):
    """
    Creates the predictions for each of the images
    :param model: The model used to obtain the predictions
    :param threshold: The threshold used for
    :param model_name: The name of the model used (for saving purposes)
    :return: Returns a prediction
    """
    if names == []:
        print('There are no ' + ' files available \n')
        return []

    output_folder = path_prediction + '/PRED ' + model_name
    test_folder = output_folder + 'Other data' #Always external data is a test data
    threshold_folder = test_folder + '/Threshold ' + str(threshold)

    if not (exists(output_folder)):
        makedirs(output_folder)
    if not (exists(test_folder)):
        makedirs(test_folder)
    if not (exists(threshold_folder)):
        makedirs(threshold_folder)

    for name in names:
        produce_prediction_files(name,path_data, model, threshold, threshold_folder)

def produce_prediction_files(name,path_data, model, threshold, threshold_folder):
    """
    This function takes an image with certain name and makes a prediction with the specified model. The prediction
    is then thresholded to obtain a boolean mask which is used to produce prediction files which are then stored in
    the threshold folder
    :param name: Name of the input image
    :param model: Model used to produce the prediction
    :param threshold: Threshold to produce the boolean mask from the prediction
    :param threshold_folder: Where the prediction files will be stored
    :return: Saves the predictions in the corresponding folder. The predictions are:
    _FIB: The original ground truth file
    _PROB: The raw prediction obtained from the model each pixel represents the probability of predicting a fiber
    _PRED: The boolean mask prediction
    _MASKED: The original image with a mask that represents TP,FP,FN
    _STAT_MASK: The mask with the TP,FP,FN
    """
    img, label = load_image_label(path_data, name)
    sizex = img.shape[0];
    sizey = img.shape[1]
    colored_image = np.concatenate((img.reshape(sizex, sizey, 1),) * 3, axis=2)
    # Reshape the image so it can be fed to the model
    img = img.reshape(1, sizex, sizey, 1)
    # Make the prediction and reshape it
    prediction = model.predict(img).reshape(sizex, sizey)
    prediction_bool = (prediction >= threshold)
    prediction = prediction * 255
    __, __, __, colored_mask = prediction_masks(prediction_bool, label)

    # Write the created images in the corresponding folder
    cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_FIB' + '.tiff'), label)
    cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_PROB' + '.tiff'),
                prediction.astype(np.uint8))
    cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_PRED' + '.tiff'),
                prediction_bool * 255)
    cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_MASKED' + '.tiff'),
                np.multiply(colored_mask, colored_image))
    cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_STATMASK' + '.tiff'),
                colored_mask * 255)