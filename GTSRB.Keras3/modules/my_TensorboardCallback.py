
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                     Tensorboard callback
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning (FIDLE) - CNRS/MIAI/UGA
# ------------------------------------------------------------------
# JL Parouty 2023
#
# See : https://keras.io/api/callbacks/
# See : https://keras.io/guides/writing_your_own_callbacks/
# See : https://pytorch.org/docs/stable/tensorboard.html

import keras
from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback(keras.callbacks.Callback):

    def __init__(self, log_dir=None):
        '''
        Init callback
        Args:
            log_dir : log directory
        '''
        self.writer = SummaryWriter(log_dir=log_dir)


    def on_epoch_end(self, epoch, logs=None):
        '''
        Record logs at epoch end
        '''

        # ---- Records all metrics (very simply)
        #
        # for k,v in logs.items():
        #     self.writer.add_scalar(k,v, epoch)

        # ---- Records and group specific metrics
        #
        self.writer.add_scalars('Accuracy',
                                {'Train':logs['accuracy'],
                                  'Validation':logs['val_accuracy']},
                                 epoch )
        
        self.writer.add_scalars('Loss',
                                {'Train':logs['loss'],
                                  'Validation':logs['val_loss']},
                                 epoch )

