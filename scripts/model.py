import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(128, 128, 3), num_classes=29):
    # Use MobileNetV2 as the base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),  # Add batch normalization layer
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),  # Add dropout regularization
        layers.Dense(num_classes, activation='softmax')  # Change num_classes to match your dataset
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model