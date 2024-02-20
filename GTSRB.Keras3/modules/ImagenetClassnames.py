# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                         Imagenet Classes
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning (FIDLE) - CNRS/MIAI/UGA
# ------------------------------------------------------------------
# JL Parouty 2024


import os
import json

class ImagenetClassnames:

    classes_file = 'ImagenetClassnames.json'

    def __init__(self):
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        with open(f'{dir_path}/{self.classes_file}') as f:
            self.classes = json.load(f)
        print(f'Imagenet classes loaded ({len(self.classes)} classes)')


    def get(self, classes_id, top_n=2):
        top_classes = [self.classes[str(i)] for i in classes_id[-top_n:]]
        return top_classes