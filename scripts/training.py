from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

def checkpoint_callback(save_dir="./checkpoints"):
    """
    Returns a ModelCheckpoint callback to save the best model during training.
    """
    os.makedirs(save_dir, exist_ok=True)
    return ModelCheckpoint(
        os.path.join(save_dir, "best_model.keras"),  # Change extension to .keras
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )

def early_stopping_callback(patience=5):
    """
    EarlyStopping callback to stop training after patience epochs of no improvement.
    """
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )


def train_with_progress(model, train_generator, valid_generator, epochs, checkpoint_filepath, early_stopping_patience=3):
    """
    Train the model with progress bars using train and validation data generators,
    and save checkpoints after each epoch. Also includes early stopping.

    Args:
        model (tf.keras.Model): The model to train.
        train_generator (DirectoryIterator): The training data generator.
        valid_generator (DirectoryIterator): The validation data generator.
        epochs (int): Number of training epochs.
        checkpoint_filepath (str): File path for saving the model checkpoints.
        early_stopping_patience (int): Number of epochs with no improvement before stopping.
    """

    # Set GPU if available
    if tf.config.list_physical_devices('GPU'):
        device = '/GPU:0'  # Use the first GPU
        print("Using GPU for training.")
    else:
        device = '/CPU:0'  # Fall back to CPU if no GPU is available
        print("Using CPU for training.")

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks for early stopping and checkpoint saving
    callbacks = [
        checkpoint_callback(checkpoint_filepath),
        early_stopping_callback(patience=early_stopping_patience)
    ]

    # Train the model with progress bars
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=callbacks,
        verbose=1  # Show progress bar
    )

    return history