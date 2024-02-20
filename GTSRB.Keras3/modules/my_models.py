
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                         Some nice models
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning (FIDLE) - CNRS/MIAI/UGA
# ------------------------------------------------------------------
# JL Parouty 2023


import keras

# ------------------------------------------------------------------
# -- A simple model, for 24x24 or 48x48 images                    --
# ------------------------------------------------------------------
#
def get_model_01(lx,ly,lz):
    
    model = keras.models.Sequential()

    model.add( keras.layers.Input((lx,ly,lz)) )
    
    model.add( keras.layers.Conv2D(96, (3,3), activation='relu' ))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Conv2D(192, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.2))

    model.add( keras.layers.Flatten()) 
    model.add( keras.layers.Dense(1500, activation='relu'))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Dense(43, activation='softmax'))
    return model
    

# ------------------------------------------------------------------
# -- A more sophisticated model, for 48x48 images                 --
# ------------------------------------------------------------------
#
def get_model_02(lx,ly,lz):
    model = keras.models.Sequential()
    
    model.add( keras.layers.Input((lx,ly,lz)) )
    
    model.add( keras.layers.Conv2D(32, (3,3),   activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add( keras.layers.MaxPooling2D((2, 2)))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Flatten()) 
    model.add( keras.layers.Dense(1152, activation='relu'))
    model.add( keras.layers.Dropout(0.5))

    model.add( keras.layers.Dense(43, activation='softmax'))
    return model



def get_model(name, lx,ly,lz):
    '''
    Return a model given by name
    Args:
        f_name : function name to retreive model
        lxly,lz : inpuy shape 
    Returns:
        model
    '''
    if name=='model_01' : return get_model_01(lx,ly,lz)
    if name=='model_02' : return get_model_01(lx,ly,lz)
    print('*** Model not found : ', name)
    return None

# A More fun version ;-)
def get_model2(name, lx,ly,lz):
    get_model=globals()['get_'+name]
    model=get_model(lx,ly,lz)
    return model



print('Module my_models loaded.')