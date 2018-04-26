import numpy as np
import matplotlib.pyplot as plt
from os import listdir, makedirs, remove
from os.path import join, exists
import cv2
from Asbestos_Utils import load_image_label, load_image, get_file_extension, load_names


def prediction_masks(prediction, label):
    """
    The input is a prediction image and a labeled image. The image matrices should be 0 where there is a fiber
    :param prediction: Image prediction with 0 if there is a fiber (can be boolean or not as long as this requirement is added)
    :param label: Boolean image prediction with 0 if there is a fiber
    :return: Pixel values for TP,FP,FN. Precision and recall for the prediction is also returned.
    colored_mask: A mask that incorporates a TP,FP,FN mask for later coloring images.
    """
    tp_mask = np.logical_not((prediction == 0) * (label == 0)) * 1
    fp_mask = np.logical_not((prediction == 0) * (label != 0)) * 1
    fn_mask = np.logical_not((prediction != 0) * (label == 0)) * 1
    colored_mask = np.ones((label.shape[0], label.shape[1], 3))
    colored_mask[:, :, 0] = tp_mask
    colored_mask[:, :, 1] = fp_mask
    colored_mask[:, :, 2] = fn_mask
    tp_pix = np.sum(tp_mask)
    fp_pix = np.sum(fp_mask)
    fn_pix = np.sum(fn_mask)
    precision = tp_pix / (tp_pix + fp_pix)
    recall = tp_pix / (tp_pix + fn_pix)
    return tp_pix, fp_pix, fn_pix, colored_mask, precision, recall


class AsbestosPredictions(object):
    def __init__(self, path_data, path_data_files, data):
        self.path_data = path_data  # Where the images are stored
        self.path_data_files = path_data_files
        self.data = data
        self.names, self.folder = self.read_data_names_folder()
        self.

    def read_data_names_folder(self):
        """
        This function reads the data names from the corresponding files
        :return: Returns the names of the data in the corresponding folder together with the folder name
        """
        '''
        This function reads the data names from the corresponding files
        '''
        if self.data == 'Train':
            names = load_names(self.path_data_files, 'Train Data')
            folder = '/Train Predictions'
        elif self.data == 'Validation':
            names = load_names(self.path_data_files, 'Validation Data')
            folder = '/Validation Predictions'
        elif self.data == 'Test':
            names = load_names(self.path_data_files, 'Test Data')
            folder = '/Test Predictions'
        else:
            print("Incorrect data type")
            names = []
            folder = []
        return names, folder








    class ImagePrediction(object):
        def __init__(self,path,name,model,threshold):
            self.name = name
            self.image,self.label = load_image_label(path,name)
            self.size = self.image.shape[0]
            self.contours_label = self.get_contours(self.label)



        def get_contours(self, image):
            """
            A function that obtains the contours for an image
            :return: A list with the contours for an image
            """
            image = image.astype(np.uint8)
            __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            __, contours_per_image, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            return contours_per_image

        def create_processed_prediction(self,model,threshold):
            prediction_unprocessed = model.predict(self.image.reshape(1,self.size,self.size, 1)).reshape(self.size,self.size)
            prediction_prob_mask = prediction_unprocessed * 255
            prediction_processed = self.postprocessing(prediction_unprocessed, threshold)
            return prediction_processed, prediction_prob_mask



        class Prediction(object):
            def __init__(self,image,model,threshold,percentage):
                self.prediction_processed, self.prediction_prob_mask = self.create_processed_prediction()
                self.prediction_contours = self.get_contours(self.prediction_processed)
            def postprocessing(self, prediction_unprocessed, threshold):
                """
                This function takes an unprocessed prediction image and a threshold and produces
                :param prediction_unprocessed:
                :param threshold:
                :return:
                """
                prediction_bool_unprocessed = (prediction_unprocessed >= threshold)
                prediction_unprocessed = prediction_bool_unprocessed * 255
                # Postprocess the predictions to remove tiny pixel arrays
                prediction_processed, contours_prediction_processed = self.remove_trash(prediction_unprocessed)
                return prediction_processed, contours_prediction_processed

            def remove_trash(self, prediction):
                """
                This function takes a prediction along with its contours and returns an image without small isolated pixels
                :return: ndarray with the prediction striped out of small isolated pixels
                """
                contours_preprocessed = self.get_contours(prediction)
                for num_contour, contour in enumerate(contours_preprocessed):
                    if int(cv2.contourArea(contour)) <= 10:
                        prediction = cv2.drawContours(prediction, contour, -1, 255, -1)
                return prediction








































    def create_predictions_test(self, model, threshold, percentage, model_name='Generic'):
        """
        Creates the predictions for each of the images
        :param model: The model used to obtain the predictions
        :param threshold: The threshold used for
        :param model_name: The name of the model used (for saving purposes)
        :return: Returns a prediction
        """
        threshold_folder = self.create_folders(model_name,threshold)

        for name in self.names:
            image, label, contours_label, size = self.image_label_info(name)
            prediction_processed, contours_prediction_processed, prediction_probability_mask = self.prediction_and_postprocessing(
                image, model, threshold)



    def image_label_prediction(self, name, model, threshold):

        image,label,contours_label,size = self.image_label_info(name)

        # Make the prediction and reshape it
        prediction_processed, contours_prediction_processed, prediction_probability_mask = self.prediction_and_postprocessing(image,model,threshold)

        return image,label,size,contours_label,prediction_probability_mask,prediction_processed,contours_prediction_processed

    def image_label_info(self,name):
        image,label = load_image_label(self.path_data,name)
        size = image.shape[0]
        contours_label = self.get_contours(label)
        return image,label,contours_label,size

    def prediction_and_postprocessing(self,image,model,threshold):
        size = image.shape[0]
        prediction_unprocessed = model.predict(image.reshape(1, size, size, 1)).reshape(size, size)
        prediction_probability_mask = prediction_unprocessed * 255
        prediction_processed, contours_prediction_processed = self.postprocessing(prediction_unprocessed, threshold)
        return prediction_processed,contours_prediction_processed,prediction_probability_mask












        tp_pix, fp_pix, fn_pix, colored_mask, precision, recall = prediction_masks(prediction_processed, label)  # Colored prediction mask

        # Write the created images in the corresponding folder
        self.save_images(name,threshold_folder,label,prediction_probability_mask,prediction_processed,colored_mask,colored_image,contours_prediction_processed)
        self.size = sizex

        objects_identified = len(contours_prediction_processed)
        total_fibers_image = len(contours_label)

        return contours_prediction_processed, contours_label, colored_mask, tp_pix, fp_pix, fn_pix, precision, recall, objects_identified, total_fibers_image






















    def fiber_discriminator_per_image(self,name,prediction, contours_label, percentage):
        found_fibers_image = 0
        image_percentages = []
        image_fiber_pixels = []
        image_intersection_pixels = []
        print("Analyzing image %s ...\n" % name)
        for num_fiber,contour_fiber in enumerate(contours_label):
            print("\t Analyzing fiber %d ...\n" % num_fiber)
            found_fiber, percentage_found, intersection_pixels, fiber_pixels = self.fiber_discriminator(prediction, contour_fiber, percentage)
            found_fibers_image += found_fiber
            image_percentages.append(percentage_found)
            image_intersection_pixels.append(intersection_pixels)
            image_fiber_pixels.append(fiber_pixels)
        return image_percentages, found_fibers_image, image_fiber_pixels, image_intersection_pixels

    def fiber_discriminator(self, prediction,contour_fiber, percentage):
        # Initialize the label mask for each fiber
        mask_label = np.zeros((self.size, self.size))
        mask_label = cv2.drawContours(mask_label, contour_fiber, -1, 255, -1)
        fiber_pixels = int(np.sum(mask_label) / 255)


        # Carry out the comparison between the ground truth and the prediction
        comparison = np.zeros((self.size, self.size))
        cv2.bitwise_and(mask_label, prediction, comparison)
        intersection_pixels = int(np.sum(comparison) / 255)
        percentage_found = intersection_pixels / fiber_pixels
        print("\t %d out of %d pixels were found: %d%%\n" % (intersection_pixels, fiber_pixels, percentage_found * 100))
        # Identify if a fiber was trully found
        if percentage_found >= percentage:
            found_fiber = 1
            print("Fiber found\n")
        else:
            found_fiber = 0
            print("Fiber not found\n")
        return found_fiber, percentage_found * 100, intersection_pixels, fiber_pixels


















    def save_images(self,name,image,threshold_folder,label,prediction_probability_mask,prediction_processed,colored_mask,colored_image,contours_prediction_processed):
        """
        Saves the predictions in the corresponding folder. The predictions are:
        _FIB: The original ground truth file
        _PROB: The raw prediction obtained from the model each pixel represents the probability of predicting a fiber
        _PRED: The boolean mask prediction
        _MASKED: The original image with a mask that represents TP,FP,FN
        _STAT_MASK: The mask with the TP,FP,FN
        _OBJNUM: Numbered objects in the prediction
        :param name: Name of the image
        :param threshold_folder: Folder where it will be stored
        :param label: Label image
        :param prediction_probability_mask: Output of the model prediction to create an image with it
        :param prediction_processed: Prediction after removing small isolated pixels
        :param colored_mask: Colored mask with TP,FP,FN
        :param colored_image: Image for use together with colored mask
        :param contours_prediction_processed: Contours of the prediction after removing small isolated pixels
        :return:
        """
        colored_image = np.concatenate((image.reshape(self.size, self.size, 1),) * 3, axis=2)
        cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_FIB' + '.tiff'), label)
        cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_PROB' + '.tiff'),
                    prediction_probability_mask.astype(np.uint8))
        cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_PRED' + '.tiff'),
                    prediction_processed)
        cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_MASKED' + '.tiff'),
                    np.multiply(colored_mask, colored_image))
        cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_STATMASK' + '.tiff'),
                    colored_mask * 255)
        cv2.imwrite(threshold_folder + '/' + name.replace(get_file_extension(name), '_OBJNUM' + '.tiff'),
                    self.numbered_fibers(np.multiply(colored_mask, colored_image), contours_prediction_processed))

    def numbered_fibers(self, colored_image,contours_prediction_processed):
        """
        This function creates an image with the objects that the model has found numbered
        :param save_path: Where should these new images should be included
        :return:
        """
        # For each image get its contours and write the area in a new image
        for num_contour, contour in enumerate(contours_prediction_processed):
            colored_image = cv2.putText(colored_image, str(num_contour), (contour[0][0][0], contour[0][0][1]),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1)
        numbered_image = colored_image
        return numbered_image


    def create_folders(self,model_name,threshold):
        output_folder = self.path_data_files + '/PRED ' + model_name
        test_folder = output_folder + self.folder
        threshold_folder = test_folder + '/Threshold ' + str(threshold)

        if not (exists(output_folder)):
            makedirs(output_folder)
        if not (exists(test_folder)):
            makedirs(test_folder)
        if not (exists(threshold_folder)):
            makedirs(threshold_folder)
        return threshold_folder


