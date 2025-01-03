from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(data_dir, batch_size=16, target_size=(128, 128), validation_split=0.2):
    """
    Prepare training and validation datasets with data augmentation applied only to the training set.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the generators.
        target_size (tuple): Target size for the images (height, width).
        validation_split (float): Fraction of data to use for validation.

    Returns:
        train_generator, valid_generator: Data generators for training and validation sets.
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,  # Set validation split
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data generator without augmentation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split  # Set validation split
    )

    # Train generator (80% of data)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'  # Use 80% of the data
    )

    # Validation generator (20% of data)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'  # Use 20% of the data
    )

    return train_generator, valid_generator