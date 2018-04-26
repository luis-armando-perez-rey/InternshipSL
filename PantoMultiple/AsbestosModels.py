######################################### CLASS ASBESTOS MODELS ############################################
from os import listdir
from os.path import join
import numpy as np
from UNET_Model import UNET


class AsbestosModels(object):
    def __init__(self, path):
        '''Initialization'''
        self.path = path
        # Read all the possible models
        self.models = list([f for f in listdir(path) if ('.h5' in f)])
        self.parameters = self.find_parameters()
        self.print_available_models()

    def find_parameters(self):
        # L: Number of layers
        # BT: Batch Size Train
        # BV: Batch Size Validation
        # E: Epochs
        # D: Data Amount
        # MD: Modified Data either 0 or 1
        # S: Size of input images in pixels
        parameters = []
        for model in self.models:
            p_arr = np.array([int(s) for s in model.split() if s.isdigit()])  # Parameter array
            if len(p_arr) < 5:
                print('The model %s does not have an appropriate format for parameter identification' % model)
                parameters.append(None)
            elif len(p_arr) == 5:
                parameters.append({'L': p_arr[0], 'BT': p_arr[1], 'BV': p_arr[2], 'E': p_arr[3], 'D': p_arr[4]})
            elif len(p_arr) == 7:
                parameters.append(
                    {'L': p_arr[0], 'BT': p_arr[1], 'BV': p_arr[2], 'E': p_arr[3], 'D': p_arr[4], 'MD': p_arr[5],
                     'S': p_arr[6]})
            else:
                parameters.append(
                   {'L': p_arr[0], 'BT': p_arr[1], 'BV': p_arr[2], 'E': p_arr[3], 'D': p_arr[4], 'HEE': p_arr[5], 'HEPI': p_arr[6],'BN':p_arr[7],'DPT':p_arr[8],'DPP':p_arr[9]})

        return parameters

    def print_available_models(self):
        for i in range(len(self.models)):
            print("Model number %d : %s : parameters" % (i, self.models[i].split('L')[0]), self.parameters[i])

    def extract_model(self, model_number,channels):
        try:
            # If the model has batch normalization include it
            batch_normalization = self.parameters[model_number]['BN']
            model = UNET(classes=2, initial_features=32, num_layers=self.parameters[model_number]['L'],
                         channels=channels,batch_normalization = bool(batch_normalization))
            model.load_weights(join(self.path, self.models[model_number]))
            return model, self.models[model_number].replace(' .h5', '')
        except KeyError:
            # If the model does not have batch normalization do not include it
            model = UNET(classes=2, initial_features=32, num_layers=self.parameters[model_number]['L'],channels=channels,batch_normalization = False)
            model.load_weights(join(self.path, self.models[model_number]))
            return model, self.models[model_number].replace(' .h5', '')
