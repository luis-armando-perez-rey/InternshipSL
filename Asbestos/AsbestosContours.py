from os import listdir, remove
from os.path import join

from Asbestos_Utils import load_image,get_file_extension
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess


class AsbestosContours(object):
    def __init__(self, path):
        self.path = path
        self.names = self.get_names('_PRED.tiff')

        self.image_size = self.get_image_size()
        self.ground_truth_names = self.get_names('_FIB.')
        self.statistics_names = self.get_names('_STATMASK.tiff')
        self.images_contours = self.get_contours(self.names)
        self.label_contours = self.get_contours(self.ground_truth_names)
        self.images_areas = self.get_areas()
        self.classification = self.get_classification()
        self.fiber_count = self.get_fibercount()
        self.remove_trash()
        self.names_refined = self.get_names('_PRED2.tiff')
        self.refined_contours = self.get_contours(self.names_refined)
        #self.precision,self.recall = self.get_statistics()

    def get_names(self, suffix):
        """
        Function that returns the file names with certain suffix
        :param suffix: The file suffix present in the desired file names
        :return: Returns a list with the filenames that include the respective suffix
        """
        names = [f for f in listdir(self.path) if (suffix in f)]
        return names

    def get_image_size(self):
        img = load_image(self.path, self.names[-1])
        return img.shape[0]

    def get_contours(self, names):
        """
        A function that obtains the contours for each of the prediction images
        :return: A list with the contours for each of the images
        """
        images_contours = []
        label_contours = []
        for num_name, name in enumerate(names):
            img = load_image(self.path, name)
            img = img.astype(np.uint8)
            __, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            __, contours_per_image, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            images_contours.append(contours_per_image)

        return images_contours

    def get_areas(self):
        """
        A function that obtains the areas for each of the contours obtained by get_contours
        :return: Returns a list with the areas of the contours for each of the images
        """
        images_areas = []
        for num_contours_per_image, contours_per_image in enumerate(self.images_contours):
            areas_per_image = np.zeros(len(contours_per_image))
            for num_contour, contour in enumerate(contours_per_image):
                areas_per_image[num_contour] = int(cv2.contourArea(contour))
            areas_per_image = areas_per_image.astype(np.int)
            images_areas.append(areas_per_image)
        return images_areas

    def get_classification(self):
        """
        A function that classifies each of the contours in an image with respect to the areas obtained previously
        :return: Returns a list with the classification of the contour areas for each of the images
        The possible contour sizes are 0,1,2,3 with 0 the smallest fibers and 3 with the biggest fibers
        """
        classification = []
        for num_areas_per_image, areas_per_image in enumerate(self.images_areas):
            classification_per_image = np.zeros(len(areas_per_image))
            for num_area, area in enumerate(areas_per_image):
                if area <= 10:
                    classification_per_image[num_area] = 0
                elif area > 10 and area <= 100:
                    classification_per_image[num_area] = 1
                elif area > 100 and area <= 200:
                    classification_per_image[num_area] = 2
                else:
                    classification_per_image[num_area] = 3
            classification.append(classification_per_image)
        return classification

    def get_fibercount(self):
        """
        A function that counts the number of contours per image with each of the classification types
        :return: Returns a list of dicts that counts how many fibers exist of each of the classification types
        """
        fiber_count = [dict(zip(range(4), np.zeros(4)))] * len(self.names)
        for num_image_class, image_class in enumerate(self.classification):
            unique, counts = np.unique(image_class, return_counts=True)
            fiber_count[num_image_class] = dict(zip(range(4), counts))
        return fiber_count

    def save_numbered_fibers(self, save_path):
        """
        This function saves the prediction images. For each of the contours a number with its area will appear beside.
        :param save_path: Where should these new images should be included
        :return:
        """
        # For each image get its contours and write the area in a new image
        for num_name, name in enumerate(self.names):
            img = load_image(self.path, name)
            contours_per_image = self.images_contours[num_name]
            for num_contour, contour in enumerate(contours_per_image):
                cv2.putText(img, str(self.images_areas[num_name][num_contour]), (contour[0][0][0], contour[0][0][1]),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=4, thickness=1)
            cv2.imwrite(join(save_path, name.replace('PRED', 'AREA')), img)

    def remove_trash(self):
        """
        This function produces a new file PRED2 that removes contours which are too small.
        :return:
        """
        for num_name, name in enumerate(self.names):
            img = load_image(self.path, name)
            contours_per_image = self.images_contours[num_name]
            mask = np.zeros(img.shape, dtype="uint8")
            for num_contour, contour in enumerate(contours_per_image):
                if self.classification[num_name][num_contour] == 0:
                    img = cv2.drawContours(img, contour, -1, 255, -1)
            cv2.imwrite(join(self.path, name), img)

    def identify_fibers(self, percentage):
        """
        This function returns the number of fibers who have been predicted according to the number of pixels from them
        found
        :return: Number of fibers found
        """


    def fiber_discriminator_all_images(self, percentage,refined = True):
        total_fibers = []
        total_found = []
        image_fiber_percentage = []
        if refined:
            names = self.names_refined
            contours = self.refined_contours
        else:
            names = self.names
            contours = self.images_contours
        for num_image, name in enumerate(names):
            image_percentages, total_fibers_image, found_fibers_image,image_fiber_pixels,image_intersection_pixels = self.fiber_discriminator_per_image(
                contours, num_image, percentage)
            total_fibers.append(total_fibers_image)
            total_found.append(found_fibers_image)
            image_fiber_percentage.append(image_percentages)
        return total_fibers,total_found,image_fiber_percentage

    def fiber_discriminator_per_image(self, contours, num_image, percentage):
        found_fibers_image = 0
        total_fibers_image = len(self.label_contours[num_image])
        image_percentages = []
        image_fiber_pixels = []
        image_intersection_pixels = []
        black_mask_prediction = np.zeros((self.image_size, self.image_size))
        black_mask_prediction = cv2.drawContours(black_mask_prediction, contours[num_image], -1, 255, -1)
        print("Analyzing image %s ...\n" % num_image)
        for num_fiber in range(total_fibers_image):
            found_fiber, percentage_found,fiber_pixels,intersection_pixels = self.fiber_discriminator(black_mask_prediction, num_image, num_fiber,
                                                                     percentage)
            found_fibers_image += found_fiber
            image_percentages.append(percentage_found)
            image_fiber_pixels.append(fiber_pixels)
            image_intersection_pixels.append(intersection_pixels)
        print("Extra objects found %d\n\n" % (len(contours[num_image]) - found_fibers_image))

        return image_percentages, total_fibers_image, found_fibers_image,image_fiber_pixels,image_intersection_pixels

    def fiber_discriminator(self, black_mask_prediction, num_image, num_fiber, percentage):
        # Initialize the label mask for each fiber
        black_mask_label = np.zeros((self.image_size, self.image_size))
        black_mask_label = cv2.drawContours(black_mask_label, self.label_contours[num_image], num_fiber, 255, -1)
        comparison = np.zeros((self.image_size, self.image_size))

        print("\t Analyzing fiber %d ...\n" % num_fiber)
        # Carry out the comparison between the ground truth and the prediction
        cv2.bitwise_and(black_mask_label, black_mask_prediction, comparison)

        fiber_pixels = int(np.sum(black_mask_label) / 255)
        # print(np.unique(comparison))
        intersection_pixels = int(np.sum(comparison) / 255)
        percentage_found = intersection_pixels / fiber_pixels
        print("\t %d out of %d pixels were found: %d%%\n" % (intersection_pixels, fiber_pixels, percentage_found * 100))
        # Identify if a fiber was trully found
        if percentage_found >= percentage:
            found_fiber = 1
        else:
            found_fiber = 0
        return found_fiber, percentage_found * 100, fiber_pixels,intersection_pixels

    def get_statistics(self,name):
        img = cv2.imread(join(self.path,name))
        tp = np.sum(img[:, :, 0] == 0)
        fp = np.sum(img[:, :, 1] == 0)
        fn = np.sum(img[:, :, 2] == 0)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(tp, fp, fn,precision,recall)
        return precision,recall
    def get_information_image(self,percentage):
        final_precision = 0
        final_recall = 0
        final_total_list = []
        final_found_list =[]
        final_total_fibers = 0
        final_total_found = 0
        for num_image,name in enumerate(self.names):
            texname = name.replace(get_file_extension(name),'.tex')
            pdfname = name.replace(get_file_extension(name),'.pdf')
            tablename = join(self.path,name.replace(get_file_extension(name),'.png'))
            image_percentages, total_fibers_image, found_fibers_image, image_fiber_pixels, image_intersection_pixels = self.fiber_discriminator_per_image(self.images_contours, num_image, percentage)
            final_total_fibers+=total_fibers_image
            final_total_found+=found_fibers_image
            final_total_list.append(total_fibers_image)
            final_found_list.append(found_fibers_image)
            precision,recall = self.get_statistics(self.statistics_names[num_image])
            dict_information = {'Fiber':range(1,len(image_percentages)+1),'Percentage %':image_percentages,'GT Pixels':image_fiber_pixels,'Predicted Pixels':image_intersection_pixels}
            dict_information2 = {'Total Fibers': [total_fibers_image],'Found Fibers': [found_fibers_image]}
            dict_information3 = {'Precision':[precision],'Recall':[recall]}
            information1 = pd.DataFrame(dict_information,columns = ['Fiber','Predicted Pixels','GT Pixels','Percentage %'])
            information1.set_index('Fiber', inplace=True)
            information2 = pd.DataFrame(dict_information2,columns = ['Found Fibers','Total Fibers'])
            #information2.set_index('Found Fibers', inplace=True)
            information3 = pd.DataFrame(dict_information3,columns = ['Precision', 'Recall'])
            #information3.set_index('Precision', inplace=True)
            template = r'''\documentclass[preview]{{standalone}}
            \usepackage{{booktabs}}
            \begin{{document}}
            {}
            {}
            {}
            \end{{document}}
            '''
            with open(texname, 'w') as f:
                f.write(template.format(information1.to_latex(column_format = '|c|c|c|c|'),information2.to_latex(column_format = '|c|c|c|'),information3.to_latex()))
            subprocess.call(['pdflatex', texname])
            subprocess.call(['magick', 'convert', '-density', '300', pdfname, '-quality', '90', tablename])
            f.close()
            remove(texname)
            remove(pdfname)
            remove(texname.replace('.tex','.log'))
            final_precision+=precision
            final_recall+=recall
        dict_information_final1 = {'Image':self.names,'Found Fibers':final_found_list,'Total Fibers':final_total_list}
        information_final1 = pd.DataFrame(dict_information_final1,columns = ['Image','Found Fibers','Total Fibers'])
        information_final1.set_index('Image', inplace=True)



        final_precision = final_precision/len(self.names)
        final_recall = final_recall/len(self.names)
        dict_information_final2 = {'Summary':[],'Overall Precision':[final_precision],'Overall Recall':[final_recall],'Total fibers found':[final_total_found],'Total fibers':[final_total_fibers]}
        information_final2 = pd.DataFrame(dict_information_final2,columns = ['Summary','Overall Precision','Overall Recall','Total fibers found','Total fibers'])
        information_final2.set_index('Summary', inplace=True)
        template2 = r'''\documentclass[preview]{{standalone}}
                    \usepackage{{booktabs}}
                    \begin{{document}}
                    {}
                    {}
                    \end{{document}}
                    '''
        texname = "Final_summary.tex"
        pdfname = "Final_summary.pdf"
        tablename = join(self.path, "Final_summary.png")
        with open(texname, 'w') as f:
            f.write(template2.format(information_final1.to_latex(column_format='|c|c|c|c|'),
                                    information_final2.to_latex(column_format='|c|c|c|c|c|')))
        subprocess.call(['pdflatex', texname])
        subprocess.call(['magick', 'convert', '-density', '300', pdfname, '-quality', '90', tablename])
        f.close()
        remove(texname)
        remove(pdfname)
        remove(texname.replace('.tex', '.log'))

