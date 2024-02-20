# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2023 
# ------------------------------------------------------------------
# 2.0 version by Achille Mbogol Touye (EFELIA-MIAI/SIMAP¨), sep 2023

from tqdm import tqdm as _tqdm
from lightning.pytorch.callbacks import TQDMProgressBar

# Créez un callback de barre de progression pour afficher les métriques d'entraînement
class CustomTrainProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._val_progress_bar     = _tqdm()
        self._predict_progress_bar = _tqdm()
        
    def init_predict_tqdm(self):
        bar=super().init_test_tqdm()
        bar.set_description("Predicting")
        return bar

    def init_train_tqdm(self):
        bar=super().init_train_tqdm()
        bar.set_description("Training")
        return bar    

    @property
    def val_progress_bar(self):
        if self._val_progress_bar is None:
            raise ValueError("The `_val_progress_bar` reference has not been set yet.")
        return self._val_progress_bar

    @property
    def predict_progress_bar(self) -> _tqdm:
        if self._predict_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._predict_progress_bar` reference has not been set yet.")
        return self._predict_progress_bar    
    

    def on_validation_start(self, trainer, pl_module):
        # Désactivez l'affichage de la barre de progression de validation
        self.val_progress_bar.disable = True  

    def on_predict_start(self, trainer, pl_module):
        # Désactivez l'affichage de la barre de progression de validation
        self.predict_progress_bar.disable = True 