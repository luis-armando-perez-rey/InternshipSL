from os.path import exists
import numpy as np
from os import makedirs
import shutil

from Asbestos_Utils import name_list, crop_save_remove_from_list, generate_complete_flipping, save_names, \
    load_names


class DataAsbestos(object):
    """
    Class that produces a split of the data into the percentages that are introduced
    It also produces crops of the specified size.
    """

    def __init__(self, path_name, data_name, size, data_augmentation=False, train_percentage=0.63, val_percentage=0.07,
                 test_percentage=0.3,
                 file_type_load='.jpg', removal=True):

        self.path_name = path_name  # Path were the data is stored
        self.data_name = data_name  # Name given to the splitted data
        self.size = size  # Size of the input images
        self.augmentation = int(data_augmentation)

        # Percentages for train validation and test
        self.train_percentage = train_percentage  # Train percentage
        self.val_percentage = val_percentage  # Validation percentage
        self.test_percentage = test_percentage  # Test percentage

        # Get the train, validation and test names with the data_name given
        self.train_names, self.val_names, self.test_names = self.train_val_test_names()

        # Tempo
        self.temp_train = './'+self.data_name+'/Temporal Train Crop Size'+str(self.size)
        self.temp_val = './'+self.data_name+'/Temporal Val Crop Size'+str(self.size)
        self.temp_test = './'+self.data_name+'/Temporal Test Crop Size'+str(self.size)
        self.name_crop_train, self.name_crop_test, self.name_crop_val = self.create_crop(load_extension=file_type_load,
                                                                                         removal=removal)

    def train_val_test_names(self):
        """
        Separate the data into train,validation and testing.
        Saves the train,validation and test names into a single file.
        :return:
        """

        if exists('./' + self.data_name):
            print("The data set already exists. Loading the specified data ...\n")
            train_names, val_names, test_names = self.read_data_names()
            return train_names, val_names, test_names
        else:
            names = name_list(self.path_name)
            light_names = np.array([f for f in names if not ('Dark' in f)])
            dark_names = np.array([f for f in names if ('Dark' in f)])
            # Actions done to light hue images
            np.random.shuffle(light_names)
            test_light = light_names[0:int(len(light_names) * self.test_percentage)]
            val_light = light_names[
                        int(len(light_names) * self.test_percentage):int(len(light_names) * self.test_percentage) + int(
                            len(light_names) * self.val_percentage)]
            train_light = light_names[
                          int(len(light_names) * self.test_percentage) + int(len(light_names) * self.val_percentage):]
            # Actions done to dark hue images
            np.random.shuffle(dark_names)
            test_dark = dark_names[0:int(len(dark_names) * self.test_percentage)]
            val_dark = dark_names[
                       int(len(dark_names) * self.test_percentage):int(len(dark_names) * self.test_percentage) + int(
                           len(dark_names) * self.val_percentage)]
            train_dark = dark_names[
                         int(len(dark_names) * self.test_percentage) + int(len(dark_names) * self.val_percentage):]

            train_names = np.concatenate((train_light, train_dark))
            val_names = np.concatenate((val_light, val_dark))
            test_names = np.concatenate((test_light, test_dark))

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

    def read_crop_names(self):
        """
        This function reads the crop names directly from the temporal folders
        :return:
        """

        save_extension = '.tiff'
        name_crop_train = name_list(self.temp_train, extension=save_extension)
        name_crop_test = name_list(self.temp_test, extension=save_extension)
        name_crop_val = name_list(self.temp_val, extension=save_extension)
        print("Total of training crops = %d\n" % len(name_crop_train))
        print("Total of validation crops = %d\n" % len(name_crop_val))
        print("Total of test crops = %d\n" % len(name_crop_val))
        return name_crop_train, name_crop_test, name_crop_val

    def create_crop(self, load_extension='.jpg', save_extension='.tiff', removal=True):
        """
        This function creates the cropped images if necessary otherwise, it gets the cropped image names directly
        from the temporal folders.
        :param load_extension:
        :param save_extension:
        :param removal:
        :return:
        """
        # If the temporal folders already exist then read and get directly the names.
        if exists(self.temp_train) and exists(self.temp_val) and exists(self.temp_test):
            print("Cropped data already exists. Obtaining the names from the cropped images")
            name_crop_train, name_crop_test, name_crop_val = self.read_crop_names()
            return name_crop_train, name_crop_test, name_crop_val
        print("Creating the cropped data ...\n")
        # Create temporal folders with the cropped data
        if exists(self.temp_train):
            shutil.rmtree(self.temp_train)
        makedirs(self.temp_train)
        if exists(self.temp_val):
            shutil.rmtree(self.temp_val)
        makedirs(self.temp_val)
        if exists(self.temp_test):
            shutil.rmtree(self.temp_test)
        makedirs(self.temp_test)

        # Create the cropped data and save it in the appropriate temporal folder

        # Train
        crops = crop_save_remove_from_list(self.path_name, self.temp_train, self.train_names, self.size,
                                           load_extension=load_extension,
                                           save_extension=save_extension,
                                           remove=removal)
        # Produce augmented data if requested
        if self.augmentation:
            print("Generating augmented data... \n")
            generate_complete_flipping(self.temp_train, self.temp_train, data_type=save_extension)
        print("Total of training crops created = %d\n" % len(name_list(self.temp_train,extension=save_extension)))

        # Validation
        crops = crop_save_remove_from_list(self.path_name, self.temp_val, self.val_names, self.size,
                                           load_extension=load_extension,
                                           save_extension=save_extension,
                                           remove=False)
        print("Total of validation crops created = %d\n" % crops)
        # Test
        crops = crop_save_remove_from_list(self.path_name, self.temp_test, self.test_names, self.size,
                                           load_extension=load_extension,
                                           save_extension=save_extension,
                                           remove=False)
        print("Total of test crops created = %d\n" % crops)



        # Get the names of the cropped images
        name_crop_train = name_list(self.temp_train, extension=save_extension)
        name_crop_test = name_list(self.temp_test, extension=save_extension)
        name_crop_val = name_list(self.temp_val, extension=save_extension)
        print('Finished loading data')
        return name_crop_train, name_crop_test, name_crop_val
