import tensorflow as tf

def save_model(model, save_path='./saved_model'):
    """
    Save the model in the TensorFlow SavedModel format.
    """
    model.save(save_path, save_format='tf')
    print(f"Model saved to {save_path}")

def convert_to_tflite(saved_model_dir, tflite_model_path='./model.tflite'):
    """
    Convert the SavedModel to TensorFlow Lite format.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")