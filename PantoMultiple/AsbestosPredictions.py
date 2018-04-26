import numpy as np
import matplotlib.pyplot as plt
from os import listdir, makedirs, remove
from os.path import join, exists
import cv2
from Asbestos_Utils import load_image_label,load_image,get_file_extension,load_names


def prediction_masks(prediction,label):
            TP_mask = np.logical_not((prediction==0) * (label==0))*1
            FP_mask = np.logical_not((prediction==0) * (label!=0))*1
            FN_mask = np.logical_not((prediction!=0) * (label==0))*1
            colored_mask = np.ones((label.shape[0],label.shape[1],3))
            colored_mask[:,:,0]=TP_mask
            colored_mask[:,:,1]=FP_mask
            colored_mask[:,:,2]=FN_mask
            return TP_mask, FP_mask, FN_mask, colored_mask


class AsbestosPredictions(object):
    def __init__(self, path_data,path_names, data='Test'):
        self.path_data = path_data # Where the images are stored
        self.path_names = path_names
        self.data = data
        self.names,self.folder = self.read_data_names_folder()
        
    def read_data_names_folder(self):
        """
        This function reads the data names from the corresponding files
        :return: Returns the names of the data in the corresponding folder together with the folder name
        """
        '''
        This function reads the data names from the corresponding files
        '''
        if self.data == 'Train':
            names = load_names(self.path_names,'Train Data')
            folder = '/Train Predictions'
        if self.data == 'Validation':   
            names = load_names(self.path_names,'Validation Data')
            folder = '/Validation Predictions'
        if self.data == 'Test': 
            names = load_names(self.path_names,'Test Data')
            folder = '/Test Predictions'
        else:
            print("Incorrect data type")
            names = []
            folder = []
        return names,folder 
    
        
    def create_predictions_test(self,model,threshold,model_name = 'Generic'):
        """
        Creates the predictions for each of the images
        :param model: The model used to obtain the predictions
        :param threshold: The threshold used for
        :param model_name: The name of the model used (for saving purposes)
        :return: Returns a prediction
        """
        if self.names ==[]:
            print('There are no '+self.data+ ' files available \n')
            return []
        
        output_folder = self.path_names+'/PRED '+model_name
        test_folder = output_folder+self.folder
        threshold_folder = test_folder+'/Threshold '+str(threshold)

        if not(exists(output_folder)):
            makedirs(output_folder)
        if not(exists(test_folder)):
            makedirs(test_folder)
        if not(exists(threshold_folder)):
            makedirs(threshold_folder)
        
        
        for name in self.names:
            self.produce_prediction_files(name,model,threshold,threshold_folder)


    def produce_prediction_files(self,name,model,threshold,threshold_folder):
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
        img, label = load_image_label(self.path_data, name)
        sizex = img.shape[0]
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






