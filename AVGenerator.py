import tensorflow.keras
import numpy as np
class AVGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_list, batch_size = 1):
        'Initialization'
        self.mix_dim = (298, 257, 2)
        self.visual_dim = (75, 1, 1792, 2)
        self.mask_dim = (298, 257, 2, 2)
        self.batch_size = batch_size
        self.file_list = file_list
        self.indexes = np.arange(len(self.file_list))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list
        current_file_list = [self.file_list[k] for k in indexes]

        # Generate data
        [X1, X2], y = self.__data_generation(current_file_list)

        return [X1, X2], y

    def __data_generation(self, current_file_list):
        'Generates data containing batch_size samples'
        # Initialization
        mask = np.empty((self.batch_size, *self.mask_dim))
        visual= np.empty((self.batch_size, *self.visual_dim))
        mix = np.empty((self.batch_size, *self.mix_dim))

        # Generate data
        for i_file, file in enumerate(current_file_list):
            data= np.load(file,allow_pickle=True).item();
            
            mix[i_file, :, :, :] = data["mix"]
            for i_speaker in range(2):
                mask[i_file, :, :, :, i_speaker] = data["mask"][i_speaker]
                visual[i_file, :, :, :, i_speaker] = np.expand_dims(data["av_pl"][i_speaker],1)

        return [mix, visual], mask