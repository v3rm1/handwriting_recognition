"""
Created Date: May 24, 2019

Created By: varunravivarma
-------------------------------------------------------------------------------

model.py:
"""

from tensorflow import keras
import pandas as pd
import numpy as np
from time import strftime


STEP_SIZE_TRAIN = 50
STEP_SIZE_VALID = 20
STEP_SIZE_TEST = 10

input_shape = (70, 70, 3)
num_classes = 27
train = 'train'
valid = 'validation'
test = 'test'
checkpoint_path = "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
epoch = 1000



def gen_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.20))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model

def main():
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=10)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None)
    model = gen_model()
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=15, width_shift_range=0.2, height_shift_range=0.2)
    train_generator = datagen.flow_from_directory(train, target_size=(70,70), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True, seed=5)
    valid_generator = datagen.flow_from_directory(valid, target_size=(70,70), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True, seed=5)
    test_generator = datagen.flow_from_directory(test, target_size=(224, 224), color_mode="rgb", batch_size=1, class_mode=None, shuffle=False, seed=5)


    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, callbacks=[model_checkpoint, early_stop], epochs=epoch)

    model.save("models/keras"+strftime("%a_%d_%b_%y__%H%M%S")+".h5")

    pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)
    predictions = [labels[k] for k in predicted_class_indices]
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_csv("results.csv",index=False)

    return

if __name__ == "__main__":
    main()
