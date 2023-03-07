from fastapi import HTTPException

def tensorflow_prediction_exception():
    raise HTTPException(status_code = 404,
                        detail = 'Unable to get tensorflow prediction. Check that required libraries are installed and input formats are correct.')
def build_model_exception():
    raise HTTPException(status_code = 404,
                        detail = 'Unable to build model. Check that required libraries (tensorflow/tensorflow-metal and tensorflow-macos) are installed.')
def load_weights_exception():
    raise HTTPException(status_code = 404,
                        detail = '''Unable to load weights. Check that tensorflow works on your device properly.
                        See if you can execute this script without error: https://www.tensorflow.org/datasets/keras_example''')