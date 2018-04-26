from os.path import join

import keras
from keras.optimizers import Adam

from DataGeneratorAsbestos import DataGeneratorAsbestos
from UNET_Model import jaccard_coef, dice_coef, UNET
from os.path import dirname,abspath


class AsbestosNetwork(object):
    def __init__(self, DataAsbestosObj, channels,classes,
                 initial_features=32,
                 num_layers=2,
                 loss="binary_crossentropy",
                 optimizer=Adam(),
                 metrics=[jaccard_coef, dice_coef],
                 data_type='.tiff', batch_normalization = False,dropout_type = 0,dropout_p=1.0, hard_example_epochs = 0, hard_examples_per_image = 0):

        # Model characteristics 
        self.classes = classes
        self.initial_features = initial_features
        self.num_layers = num_layers
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_normalization = batch_normalization
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p
        self.channels = channels
        # Model initialization
        self.model = UNET(classes=self.classes, initial_features=self.initial_features,
                          num_layers=self.num_layers, loss=self.loss,
                          optimizer=self.optimizer, metrics=self.metrics,batch_normalization = self.batch_normalization,
                          dropout_type=self.dropout_type, dropout_p=self.dropout_p,channels = self.channels)

        # Path names
        self.path_name = DataAsbestosObj.path_name
        self.train_path = DataAsbestosObj.temp_train
        self.val_path = DataAsbestosObj.temp_val

        # Data names
        self.data_name = DataAsbestosObj.data_name
        self.train_names = DataAsbestosObj.name_crop_train
        self.val_names = DataAsbestosObj.name_crop_val

        # Network parameters
        self.data_type = data_type
        self.params = {'L': num_layers, 'S': DataAsbestosObj.size, 'MD': DataAsbestosObj.augmentation,
                       'HEE':hard_example_epochs,'HEPI': hard_examples_per_image,
                       'BN':int(batch_normalization),'DPT':int(dropout_type),'DPP':int(dropout_p*100)}

    def train_model(self, prefix_log, batch_size_train=10, batch_size_val=10, epochs=10):
        # Introduce the parameter values in the corresponding dictionary entries
        self.params['BT'] = batch_size_train
        self.params['E'] = epochs
        self.params['BV'] = batch_size_val

        # Print the parameters of the network
        print("Parameters of the model are: \n")
        print('Layers :: %d \n' % (self.params['L']))
        print('Size training batch :: %d \n' % (self.params['BT']))
        print('Size validation batch :: %d \n' % (self.params['BV']))
        print('Epochs :: %d \n' % (self.params['E']))
        print('Input size :: %d \n' % (self.params['S']))
        print('Modified data :: %d \n' % (self.params['MD']))
        parameter_string = ' L ' + str(self.params['L']) + ' BT ' + str(self.params['BT']) + ' BV ' + str(self.params['BV'])+' E ' + str(self.params['E']) +' D '+ str(len(self.train_names)) + ' MD ' + str(self.params['MD']) + ' S ' + str(self.params['S']) + ' HEE '+ str(self.params['HEE'])+ ' HEPI '+str(self.params['HEPI']) + ' BN ' + str(self.params['BN']) + ' DPT '+str(self.params['DPT'])+' DPP '+str(self.params['DPP'])

        save_folder = dirname(abspath('__file__')) +'/' + self.data_name + '/' +'TensorBoard/'+ prefix_log + '/'+parameter_string
        print('The tensorboard logfiles will be saved in {}...\n'.format(save_folder))
        self.model_name = self.data_name + ' ' + prefix_log.replace('/', '') + parameter_string
        # Parameters for the data generator of the test data
        train_params = {'path': self.train_path,
                        'file_names': self.train_names,
                        'dim_x': self.params['S'],
                        'dim_y': self.params['S'],
                        'data_type': self.data_type,
                        'batch_size': self.params['BT'],
                        'shuffle': True}
        validation_params = dict(path=self.val_path, file_names=self.val_names, dim_x=self.params['S'],
                                 dim_y=self.params['S'], data_type=self.data_type, batch_size=self.params['BV'],
                                 shuffle=True)
        print('Train data used = %d\n' % len(self.train_names))
        print('Validation data used = %d\n' % len(self.val_names))

        training_generator = DataGeneratorAsbestos(**train_params).generate()
        if validation_params['file_names'] != []:
            validation_generator = DataGeneratorAsbestos(**validation_params).generate()
        else:
            validation_generator = None

        TB = keras.callbacks.TensorBoard(write_graph=True,
                                         log_dir=save_folder,
                                         write_images=True,
                                         embeddings_freq=0)

        CP = keras.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filepath='./' + self.data_name + '/' + prefix_log.replace('/', '')+ ' BEST ' + parameter_string+ ' .h5',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1)

        self.model.fit_generator(generator=training_generator,
                                 steps_per_epoch=len(self.train_names) // self.params['BT'],
                                 epochs=self.params['E'],
                                 validation_data=validation_generator,
                                 validation_steps=len(self.val_names) // self.params['BV'],
                                 callbacks=[TB, CP])
        self.model.save_weights(('./' + self.data_name + '/' + prefix_log.replace('/', '') + parameter_string+ ' .h5'))
        print("The model with the best weights according to validation loss is loaded ...\n")
        self.model = UNET(classes=self.classes, initial_features=self.initial_features,
                          num_layers=self.num_layers, loss=self.loss,
                          optimizer=self.optimizer, metrics=self.metrics,batch_normalization = self.batch_normalization,
                          dropout_type=0, dropout_p=self.dropout_p,channels = self.channels)
        self.model.load_weights('./' + self.data_name + '/' + prefix_log.replace('/', '')+ ' BEST ' + parameter_string+ ' .h5')

