import keras
from keras.optimizers import Adam

from DataGeneratorPantographMultipleClass import DataGeneratorPantographMultipleClass
from UNET_Model import jaccard_coef, dice_coef, UNET


class PantographNetworkMultipleClass(object):
    def __init__(self, DataPantograph,
                 classes=3,
                 initial_features=32,
                 num_layers=2,
                 loss="binary_crossentropy",
                 optimizer=Adam(),
                 metrics=[jaccard_coef, dice_coef],
                 data_type='.tiff', hard_example_epochs = 0, hard_examples_per_image = 0,channels = 1):

        # Model initialization
        self.model = UNET(classes=classes, initial_features=initial_features,
                          num_layers=num_layers, loss=loss,
                          optimizer=optimizer, metrics=metrics,channels=channels)

        # Path names
        self.multiple_class = DataPantograph.multiple_class
        self.path_name = DataPantograph.path_name
        self.train_path = DataPantograph.temp_train
        self.val_path = DataPantograph.temp_val

        # Data names
        self.data_name = DataPantograph.data_name
        self.train_names = DataPantograph.train_names
        self.val_names = DataPantograph.val_names
        self.size = DataPantograph.size

        # Network parameters
        self.data_type = data_type
        self.params = {'L': num_layers,  'MD': DataPantograph.augmentation,
                       'HEE':hard_example_epochs,'HEPI': hard_examples_per_image,'SX':self.size[0],'SY':self.size[1]}

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
        print('Modified data :: %d \n' % (self.params['MD']))
        parameter_string = ' L ' + str(self.params['L']) + ' BT ' + str(self.params['BT']) + ' BV ' + str(self.params['BV'])+' E ' + str(self.params['E']) +' D '+ str(len(self.train_names)) + ' MD ' + str(self.params['MD'])  +' SX '+str(self.params['SX'])+'SY'+str(self.params['SY'])+ ' HEE '+ str(self.params['HEE'])+ ' HEPI '+str(self.params['HEPI'])

        save_folder = ('/tmp/' + prefix_log + '/'+parameter_string)
        self.model_name = self.data_name + ' ' + prefix_log.replace('/', '') + parameter_string
        # Parameters for the data generator of the test data
        train_params = {'path': self.train_path,
                        'file_names': self.train_names,
                        'size':self.size,
                        'data_type': self.data_type,
                        'batch_size': self.params['BT'],
                        'shuffle': True}
        validation_params = {'path':self.val_path,
                             'file_names':self.val_names,
                             'size':(256,1600),
                             'data_type':self.data_type,
                             'batch_size':self.params['BV'],
                             'shuffle':True}
        print('Train data used = %d\n' % len(self.train_names))
        print('Validation data used = %d\n' % len(self.val_names))

        training_generator = DataGeneratorPantographMultipleClass(**train_params).generate()
        if validation_params['file_names'] != []:
            validation_generator = DataGeneratorPantographMultipleClass(**validation_params).generate()
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
        self.model.load_weights('./' + self.data_name + '/' + prefix_log.replace('/', '')+ ' BEST ' + parameter_string+ ' .h5')

