


from os.path import isfile,join
import numpy as np
import cv2
from Asbestos_Utils import load_image


class DataGeneratorAsbestos(object):
    def __init__(self,path,file_names,dim_x,dim_y,data_type = ".jpg",batch_size = 10, shuffle = True):
        '''Initialization'''
        self.path = path # Path from which the data is read
        self.file_names = file_names # The names of the files that will be read 
        self.dim_x = dim_x # The dimensions of the data
        self.dim_y = dim_y
        self.data_type = data_type # The data type of the data that will be read
        self.batch_size = batch_size # The size of the batches that will be fed to the network
        self.shuffle = shuffle # Shuffling of the data inserted to the network
    
    def __get_exploration_order(self):
        '''
        Generates order of exploration for the data files
        '''
        # Find exploration order
        indexes = np.arange(len(self.file_names)) # Make a list of indexes for the names
        if self.shuffle == True:
            np.random.shuffle(indexes) # Shuffle the indexes and return them
        return indexes
    
    def generate(self):
        '''
        Generates the data batches that will be fed to the network. 
        '''
        while 1:
            indexes = self.__get_exploration_order() # Obtain the indexes for the data files
            imax = int(len(indexes)/self.batch_size) # Determine the number of batches per epoch
            for i in range(imax):
                # Get the names of the files for each of the batches
                batch_names = [self.file_names[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate the data from the names defined previously
                X,y = self.__data_generation(batch_names)
                yield X,y
                
    def __data_generation(self,batch_names):
        #Initialization of the batches
        X = np.empty((self.batch_size,self.dim_x,self.dim_y,1))
        # Change for using only 1 class "FIBERS" as output , last dimension to 1
        y = np.empty((self.batch_size,self.dim_x,self.dim_y,1))
        
        for i, name in enumerate(batch_names):
            X[i,:,:,0] = load_image(self.path,name) # Load the data for the image

            # Load the data for the classification images
            if isfile(join(self.path,name.replace(self.data_type,'_FIB'+self.data_type))):
                label = load_image(self.path,name.replace(self.data_type,'_FIB'+self.data_type)) # Load the image of the annotations
                label = label.astype(np.uint8) #Transform to uint8
                _,label = cv2.threshold(label,127,255,0) # Threshold the image (while reading some annotations are not with values [0,255] only)
            else:
                label = np.zeros(self.dim_x,self.dim_y,1) # Create a label matrix with all zeros if the image was not annotated (doesn't have fibers)
            y[i,:,:,0] = 1*(label==255) # Classify as 1 if the value is 255 for the "fiber" mask.
            # Change for using only 1 class "FIBERS" as output 
            # y[i,:,:,1] = 1*(label==0) # Classify as 1 if the value is 0 for the "not a fiber" mask.
        return X,y