from os.path import exists, join
import numpy as np
from os import makedirs
import shutil
from scipy.signal import convolve2d
import cv2


from Asbestos_Utils import name_list, crop_save_remove_from_list, generate_complete_flipping, save_names, \
    load_names, get_file_extension,generate_complete_flipping


def pad_image(image):
    sizex = image.shape[1]
    sizey = image.shape[0]
    extrapixX = 1600-sizex
    extrapixY = 256-sizey
    left = extrapixX//2
    right = extrapixX-left
    top = extrapixY//2
    bottom = extrapixY-top
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 0,value = [255,255,255])
    return image




class DataPantograph(object):
    """
    Class that produces a split of the data into the percentages that are introduced
    It also produces crops of the specified size.
    """

    def __init__(self, path_name, data_name,size, data_augmentation=False, train_percentage=0.63, val_percentage=0.07,
                 test_percentage=0.3,
                 load_extension='.jpg',augmentation = False):

        self.path_name = path_name  # Path were the data is stored
        self.data_name = data_name  # Name given to the splitted data
        self.augmentation = int(data_augmentation)
        self.load_extension = load_extension
        self.size = size
        self.augmentation = augmentation
        # Percentages for train validation and test
        self.train_percentage = train_percentage  # Train percentage
        self.val_percentage = val_percentage  # Validation percentage
        self.test_percentage = test_percentage  # Test percentage

        self.temp_train = './' + self.data_name + '/Temporal Train Crop Size'+str(self.size[0])+ 'x' + str(size[1])
        self.temp_val = './' + self.data_name + '/Temporal Val Crop Size'+str(self.size[0])+ 'x'+str(self.size[1])
        self.temp_test = './' + self.data_name + '/Temporal Test Crop Size'+str(self.size[0])+'x'+str(self.size[1])

        # Get the train, validation and test names with the data_name given
        self.train_names_original, self.val_names_original, self.test_names_original = self.train_val_test_names()
        self.get_chip_images(size)
        self.train_names = name_list(self.temp_train,load_extension)
        self.test_names = name_list(self.temp_test,load_extension)
        self.val_names = name_list(self.temp_val,load_extension)
    def train_val_test_names(self):
        """
        Separate the data into train,validation and testing.
        Saves the train,validation and test names into a single file.
        :return:
        """

        if exists('./' + self.data_name):
            print("The data set already exists. Loading the specified data ...\n")
            train_names, val_names, test_names = self.read_data_names()
            print('Size of training set: %d\n' % (len(train_names)))
            print('Size of validation set: %d\n' % (len(val_names)))
            print('Size of test set: %d\n' % (len(test_names)))
            return train_names, val_names, test_names
        else:
            names = name_list(self.path_name,extension=self.load_extension )

            # Actions done to light hue images
            np.random.shuffle(names)
            test_names = names[0:int(len(names) * self.test_percentage)]
            val_names = names[
                        int(len(names) * self.test_percentage):int(len(names) * self.test_percentage) + int(
                            len(names) * self.val_percentage)]
            train_names = names[int(len(names) * self.test_percentage) + int(len(names) * self.val_percentage):]

            print('Size of training set: %d\n' % (len(train_names)))
            print('Size of validation set: %d\n' % (len(val_names)))
            print('Size of test set: %d\n' % (len(test_names)))

            makedirs('./' + self.data_name)
            save_names(train_names, './' + self.data_name, 'Train Data')
            save_names(val_names, './' + self.data_name, 'Validation Data')
            save_names(test_names, './' + self.data_name, 'Test Data')
            return train_names, val_names, test_names

    def read_data_names(self):
        """
        Separate the data into train,validation and testing.
        Saves the train,validation and test names into a single file.
        :return:
        """
        train_names = load_names('./' + self.data_name, 'Train Data')
        val_names = load_names('./' + self.data_name, 'Validation Data')
        test_names = load_names('./' + self.data_name, 'Test Data')
        return train_names, val_names, test_names


    def get_chip_images(self,size):
        name_array = [self.train_names_original,self.val_names_original,self.test_names_original]
        folder_array = [self.temp_train,self.temp_val,self.temp_test]
        # Train data creation
        names = name_array[0]
        if not(exists(folder_array[0])):
            makedirs(folder_array[0])
            for name in names:
                image = cv2.imread(join(self.path_name,name))
                label = cv2.imread(join(self.path_name,name.replace(get_file_extension(name),'_FIB'+get_file_extension(name))),0)
                self.convolutional_selection(folder_array[0], size, image, label, name)
        if self.augmentation:
            generate_complete_flipping(folder_array[0],folder_array[0], data_type=self.load_extension,y_flip= False,xy_flip=False)
        # Validation Test Data
        for num_folder in range(2):
            folder = folder_array[num_folder+1]
            names = name_array[num_folder+1]
            if not(exists(folder)):
                makedirs(folder)
            for name in names:
                image = cv2.imread(join(self.path_name, name))
                label = cv2.imread(join(self.path_name, name.replace(get_file_extension(name), '_FIB' + get_file_extension(name))), 0)
                image = pad_image(image)
                label = pad_image(label)
                cv2.imwrite(join(folder, name), image)
                cv2.imwrite(join(folder,
                                 name.replace(get_file_extension(name),
                                 '_FIB' + get_file_extension(name))),label)

    def convolutional_selection(self,save_path, size, img, label, name):
        # Obtain the left top corner for the hard example window
        chip_names = []
        # Convolve the label to find the relevant spots
        print(self.gaussian_kernel(10).shape)
        print(label.shape)
        __, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY_INV)
        convolved = convolve2d(label, self.gaussian_kernel((np.min(self.size))), mode='valid')
        __, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY_INV)
        while np.sum(convolved) != 0:
            index1, index2 = np.unravel_index(convolved.argmax(), convolved.shape)
            chip_image = img[index1:(index1 + size[0]),
                         index2:(index2 + size[1])]  # Get the hard example window
            chip_label = label[index1:(index1 + size[0]),
                         index2:(index2 + size[1])]  # Get the hard example label window

            # Remove a certain search area for the convolution
            index1_rmv_conv_start = max([index1 - int(size[0] / 4) + 1, 0])
            index2_rmv_conv_start = max([index2 - int(size[1] / 4) + 1, 0])
            index1_rmv_conv_end = min([index1_rmv_conv_start + (size[0] - 2), img.shape[0]])
            index2_rmv_conv_end = min([index2_rmv_conv_start + (size[1] - 2), img.shape[1]])
            convolved[index1_rmv_conv_start:index1_rmv_conv_end, index2_rmv_conv_start:index2_rmv_conv_end] = 0

            save_name = name.replace(get_file_extension(name),
                                     '_H_' + str(index1) + '_' + str(index2) + get_file_extension(name))
            cv2.imwrite(join(save_path, save_name), chip_image)
            cv2.imwrite(join(save_path,
                             save_name.replace(get_file_extension(save_name), '_FIB' + get_file_extension(save_name))),
                        chip_label)
        return chip_names

    def gaussian_kernel(self, sigma):
        sizex = self.size[1]
        sizey = self.size[0]
        x, y = np.mgrid[-sizex/2:sizex/2, -sizey/2:sizey/2]
        g = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma/2) ** 2))
        return g / g.sum()

