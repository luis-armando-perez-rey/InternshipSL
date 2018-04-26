import subprocess
from os import makedirs, remove
from os.path import join, exists
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from Asbestos_Utils import load_image_label, get_file_extension, load_names


def get_contours(image, invert = True):
    """
    A function that obtains the contours from an image
    :return: A list with the contours for an image
    """
    image = image.astype(np.uint8)
    if invert:
        __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        __, contours_per_image, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        __, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        __, contours_per_image, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours_per_image


class AsbestosModelProcess(object):
    def __init__(self, path_data, path_data_files, model, model_name, threshold, percentage, data='Validation'):
        self.path_data = path_data  # Where the images are stored
        self.path_data_files = path_data_files  # Where the data files that contain the names in the data are stored
        self.data = data  # The type of data that will be processed
        self.names, self.folder = self.read_data_names_folder()
        self.threshold_folder = self.create_folders(model_name, threshold)
        self.total_precision = 0
        self.total_recall = 0
        self.total_found_fibers = 0
        self.total_fibers_all = 0
        self.total_false_positive_objects = 0
        self.model_name = model_name
        self.summary_found_fibers_array = []
        self.summary_total_fibers_array = []
        self.summary_percentage_array = []
        self.summary_precision = []
        self.summary_recall = []
        for name in self.names:
            ImageObject = self.ImagePrediction(self.path_data, name, model, threshold, percentage)
            #ImageObject.prediction_object.save_images(self.threshold_folder)
            precision, recall, found_fibers, total_fibers, image_percentages,false_positive_objects = ImageObject.prediction_object.summary_image(
                self.threshold_folder)
            self.total_false_positive_objects += false_positive_objects
            self.total_precision += precision
            self.total_recall += recall
            self.total_found_fibers += found_fibers
            self.total_fibers_all += total_fibers

            self.summary_precision.append(precision)
            self.summary_recall.append(recall)
            self.summary_found_fibers_array.append(found_fibers)
            self.summary_total_fibers_array.append(total_fibers)
            self.summary_percentage_array.append(100 * found_fibers / total_fibers)
        self.total_precision = self.total_precision / len(self.names)
        self.total_recall = self.total_recall / len(self.names)
        self.summary_data()
        print(self.summary_found_fibers_array)

    def summary_data(self):
        """
        Produces an image with the summary of the data obtained previously
        :return: Saves in the threshold_folder an image with the corresponding produced table
        """
        try:
            object_precision = self.total_found_fibers/(self.total_found_fibers+self.total_false_positive_objects)
        except ZeroDivisionError:
            object_precision = None
        try:
            object_recall = self.total_found_fibers/self.total_fibers_all
        except ZeroDivisionError:
            object_recall = None

        file_images_summary ='IMGS_SUMMARY'+'.txt'
        fis = open(join(self.threshold_folder,file_images_summary),'w+')
        for num_names in range(len(self.names)):
            fis.write('Image {} Found_fibers {} Total_fibers {} Percentage {} Precision {} Recall {} \n'.format(self.names[num_names],
                        self.summary_found_fibers_array[num_names],self.summary_total_fibers_array[num_names],
                        self.summary_percentage_array[num_names],self.summary_precision[num_names],self.summary_recall[num_names]))
        fis.close()

        file_name_summary = 'SUMMARY'+'.txt'
        fns = open(join(self.threshold_folder,file_name_summary),'w+')
        fns.write('Pixel_precision {} Pixel_recall {} \n'.format(self.total_precision,self.total_recall))
        fns.write('Total_fibers {} Found_fibers {} \n'.format(self.total_fibers_all,self.total_found_fibers))
        fns.write('Object_Precision {} Object_Recall{} \n'.format(object_precision,object_recall))
        fns.close()
        print("The object precision is {} and the object recall is {}\n".format(object_precision,object_recall))

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

    def create_folders(self, model_name, threshold):
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

    ####### IMAGE CLASS #######
    class ImagePrediction(object):
        def __init__(self, path, name, model, threshold, percentage):
            self.name = name
            self.image, self.label = load_image_label(path, name)
            self.size = self.image.shape[0]
            self.contours_label = get_contours(self.label)
            self.prediction_object = self.Prediction(self.name, self.image, self.label, self.contours_label, model,
                                                     threshold, percentage)

        class Prediction(object):
            def __init__(self, name, image, label, contours_label, model, threshold_range, percentage):
                self.name = name
                self.image = image
                self.label = label
                self.size = image.shape[0]
                self.contours_label = contours_label
                self.total_fibers = len(contours_label)
                self.model = model
                self.threshold_range = threshold_range
                self.percentage = percentage
                prediction_unprocessed = self.model.predict(self.image.reshape(1, self.size, self.size, 1)).reshape(
                                         self.size, self.size)
                pixel_precision_array = []
                pixel_recall_array = []
                object_precision_array = []
                object_recall_array = []
                data_list_object = [name]
                data_list_pixel = [name]
                for threshold in threshold_range:
                    self.prediction, self.prediction_prob_mask, self.prediction_unprocessed = self.prediction_generation(prediction_unprocessed,threshold)
                    self.prediction_contours = get_contours(self.prediction)
                    self.tp_pix, self.fp_pix, self.fn_pix, self.colored_mask, self.precision, self.recall = self.prediction_masks()
                    pixel_precision_array.append(self.precision)
                    pixel_recall_array.append(self.recall)
                    self.found_fibers, self.image_percentages, self.image_fiber_pixels, self.image_intersection_pixels,self. false_positive_objects = self.fiber_identification()
                    try:
                        object_precision = self.found_fibers / (self.found_fibers + self.false_positive_objects)
                    except ZeroDivisionError:
                        object_precision = None
                    try:
                        object_recall = self.found_fibers / (self.total_fibers)
                    except ZeroDivisionError:
                        object_recall = None
                    object_precision_array.append(object_precision)
                    object_recall_array.append(object_recall)
                    data_list_object.append((object_precision,object_recall))
                    data_list_pixel.append((self.precision, self.recall))
                df_image = DataFrame([],columns)

            
            def summary_image(self, threshold_folder):
                """
                Produces an image with the summary of the data obtained previously
                :param threshold_folder: Where the image is going to be stored
                :return: Saves in the threshold_folder an image with the corresponding produced table
                """
                print("Saving the image summary table")



                try:
                    object_precision = self.found_fibers / (self.found_fibers + self.false_positive_objects)
                except ZeroDivisionError:
                    object_precision = None
                try:
                    object_recall = self.found_fibers/(self.total_fibers)
                except ZeroDivisionError:
                    object_recall = None

                filename_fiber_detailed = self.name.replace(get_file_extension(self.name),'') + '_FD.txt'
                ffd = open(join(threshold_folder,filename_fiber_detailed), 'w+')
                for fiber in range(len(self.image_percentages)):
                    ffd.write('Fiber {} Predicted_Pixels {} GTPixels {} Percentage {} \n'.format(fiber+1,self.image_intersection_pixels[fiber],
                                                                                            self.image_fiber_pixels[fiber],self.image_percentages[fiber]))
                ffd.close()

                filename_image_summary = self.name.replace(get_file_extension(self.name),'') + '_FIS.txt'
                fis = open(join(threshold_folder,filename_image_summary),'w+')

                fis.write('Total_fibers {} Found_fibers {} False_Positive_Objects {} \n'.format(self.total_fibers,self.found_fibers,self.false_positive_objects))
                fis.write('Object_precision {} Object_recall {} Pixel_precision {} Pixel_recall {} \n'.format(object_precision,object_recall,self.precision,self.recall))
                fis.close()

                return self.precision, self.recall, self.found_fibers, self.total_fibers, self.image_percentages,self.false_positive_objects


            def fiber_identification(self):
                """
                Function that identifies the fibers in an image
                :return: found_fibers(number) The amount of fibers found in the image
                image_percentages(array) an array that includes the percentage found for each fiber
                image_fiber_pixels(array) an array that includes the amount of pixels for each fiber
                image_intersection_pixels(array) an array that includes the amount of pixels found for each fiber
                """
                # Initialize the outputs
                fiber_contours = get_contours(self.label)
                found_fibers = 0
                image_percentages = []
                image_fiber_pixels = []
                image_intersection_pixels = []
                # Produce a mask for prediction comparizon
                black_mask_prediction = np.zeros((self.size, self.size))
                black_mask_prediction = cv2.drawContours(black_mask_prediction, self.prediction_contours, -1, 255, -1)
                print("Analyzing image %s ...\n" % self.name)
                # Analyze whether each of the fibers in the image has been found and how much of the fiber was
                # identified
                for num_fiber, fiber in enumerate(fiber_contours):
                    print("\t Analyzing fiber %d ...\n" % num_fiber)

                    found_fiber, percentage_found, intersection_pixels, fiber_pixels = self.fiber_discriminator(
                        fiber, black_mask_prediction)
                    # Remove the identified fiber from the prediction
                    black_mask_prediction = self.remove_fiber_from_prediction(fiber, black_mask_prediction)
                    # Include the results for each fiber in the overall output
                    found_fibers += found_fiber
                    image_percentages.append(percentage_found)
                    image_intersection_pixels.append(intersection_pixels)
                    image_fiber_pixels.append(fiber_pixels)
                false_positive_amount = self.false_positive_object_count(black_mask_prediction)
                print(false_positive_amount)
                return found_fibers, image_percentages, image_fiber_pixels, image_intersection_pixels, false_positive_amount

            def false_positive_object_count(self,black_mask_prediction):
                """
                Counts the number of false positive objects in the prediction
                :param black_mask_prediction: The mask without the fiber predictions
                :return: The number of false positive objects found
                """
                false_positive_contours = get_contours(black_mask_prediction,invert = False)
                return len(false_positive_contours)



            def fiber_discriminator(self, fiber, black_mask_prediction):
                """
                Function that takes the contours of a fiber and tells whether the fiber was identified by the prediction
                with respect to certain percentage threshold.
                :param fiber: Contour for the fiber
                :param black_mask_prediction: Prediction mask to compare with
                :return: found_fibers(number) Either 1 (found) or 0 (not found)
                percentage_found Percentage of correctly identified pixels
                fiber_pixels The amount of pixels for each fiber
                intersection_pixels The amount of pixels found by the prediction for each fiber
                """
                # Create a mask for the fiber
                mask_label = np.zeros((self.size, self.size))
                mask_label = cv2.drawContours(mask_label, [fiber], -1, 255, -1)

                fiber_pixels = int(np.sum(mask_label) / 255)  # Count of the fiber pixels
                # Compare the fiber mask with the prediction
                comparison = np.zeros((self.size, self.size))
                cv2.bitwise_and(mask_label, black_mask_prediction, comparison)
                intersection_pixels = int(np.sum(comparison) / 255)  # Count pixels in the intersection
                percentage_found = intersection_pixels / fiber_pixels  # Percentage of fiber pixels identified
                print("\t %d out of %d pixels were found: %d%%\n" % (
                intersection_pixels, fiber_pixels, percentage_found * 100))
                # Identify if a fiber was trully found
                if percentage_found >= self.percentage:
                    found_fiber = 1
                    print("Fiber found\n")
                else:
                    found_fiber = 0
                    print("Fiber not found\n")
                # Remove the analyzed fiber pixels from the prediction (for False Positive purposes)

                return found_fiber, percentage_found * 100, intersection_pixels, fiber_pixels

            def remove_fiber_from_prediction(self, fiber, black_mask_prediction):
                """
                This function removes from the prediction mask the pixels which predict certain fiber.
                :param fiber: The fiber contour that will be removed from the prediction mask
                :param black_mask_prediction: The prediction mask
                :return: Returns the modified prediction mask without the selected fiber pixels.
                """
                # Draw the fiber over the black prediction mask
                black_mask_prediction = cv2.drawContours(black_mask_prediction, [fiber], -1, 255, -1)
                # Create a fiber comparison mask
                fiber_mask = np.zeros((self.size, self.size))
                fiber_mask = cv2.drawContours(fiber_mask, [fiber], -1, 255, -1)
                # Get the contours of this new mask
                new_contours = get_contours(black_mask_prediction,invert= False)
                max_contour_intersection = 0 # Define an initial amount of pixels for the comparison
                num_contour_removal = 0 # Choose the first new contour as the one that will be removed
                comparison2 = np.zeros((self.size,self.size))
                for num_contour, contour in enumerate(new_contours):
                    # Create a mask for the comparison between the new contours and the fiber
                    black_mask_new = np.zeros((self.size, self.size))
                    black_mask_new = cv2.drawContours(black_mask_new, [contour], -1, 255, -1)
                    intersection = np.sum(cv2.bitwise_and(black_mask_new, fiber_mask, comparison2))
                    if intersection > max_contour_intersection:
                        max_contour_intersection = intersection
                        num_contour_removal = num_contour
                black_mask_prediction = cv2.drawContours(black_mask_prediction, new_contours, num_contour_removal, 0, -1)
                return black_mask_prediction

            def prediction_generation(self,prediction_unprocessed,threshold):
                """
                This function takes an unprocessed prediction image and a threshold and produces
                :param prediction_unprocessed:
                :return:
                """
                prediction_probability_mask = prediction_unprocessed * 255
                prediction_processed, prediction_unprocessed = self.postprocessing(prediction_unprocessed,threshold)
                return prediction_processed, prediction_probability_mask, prediction_unprocessed

            def postprocessing(self, prediction_unprocessed,threshold):
                """
                This function takes an unprocessed prediction image and a threshold and produces
                :param prediction_unprocessed:
                :param threshold:
                :return:
                """
                prediction_bool_unprocessed = (prediction_unprocessed >= threshold)
                prediction_unprocessed = prediction_bool_unprocessed * 255
                # Postprocess the predictions to remove tiny pixel arrays
                prediction_processed = self.remove_trash(prediction_unprocessed)
                return prediction_processed, prediction_unprocessed

            def remove_trash(self, prediction_unprocessed):
                """
                This function takes a prediction along with its contours and returns an image without small isolated pixels
                :return: ndarray with the prediction striped out of small isolated pixels
                """
                contours_preprocessed = get_contours(prediction_unprocessed)
                for num_contour, contour in enumerate(contours_preprocessed):
                    if int(cv2.contourArea(contour)) <= 18:
                        prediction_unprocessed = cv2.drawContours(prediction_unprocessed, [contour], -1, 255, -1)
                prediction_processed = prediction_unprocessed
                return prediction_processed

            def prediction_masks(self):
                """
                The input is a prediction image and a labeled image. The image matrices should be 0 where there is a fiber
                :param prediction: Image prediction with 0 if there is a fiber (can be boolean or not as long as this requirement is added)
                :return: Pixel values for TP,FP,FN. Precision and recall for the prediction is also returned.
                colored_mask: A mask that incorporates a TP,FP,FN mask for later coloring images.
                """
                tp_mask = np.logical_not((self.prediction == 0) * (self.label == 0)) * 1
                fp_mask = np.logical_not((self.prediction == 0) * (self.label != 0)) * 1
                fn_mask = np.logical_not((self.prediction != 0) * (self.label == 0)) * 1
                colored_mask = np.ones((self.label.shape[0], self.label.shape[1], 3))
                colored_mask[:, :, 0] = tp_mask
                colored_mask[:, :, 1] = fp_mask
                colored_mask[:, :, 2] = fn_mask
                tp_pix = np.sum(np.logical_not(tp_mask))
                fp_pix = np.sum(np.logical_not(fp_mask))
                fn_pix = np.sum(np.logical_not(fn_mask))
                try:
                    precision = tp_pix / (tp_pix + fp_pix)
                except ZeroDivisionError:
                    precision = None
                try:
                    recall = tp_pix / (tp_pix + fn_pix)
                except ZeroDivisionError:
                    recall = None


                return tp_pix, fp_pix, fn_pix, colored_mask, precision, recall
