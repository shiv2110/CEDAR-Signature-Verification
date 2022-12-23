from keras.models import load_model
import numpy as np

model = load_model('final_model_siamese_1.h5')

def predict (img_array):
    img_array = img_array/255.0
    test = np.expand_dims(img_array, axis = 0)
    result = model.predict(test)
    return result[0][0]