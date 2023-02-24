
import tensorflow_datasets as tfds
import argparse
import sys
from src.RSIRUtils import *


if __name__ == "__main__":
    epochs = int(sys.argv[0])
    imageSize = int(sys.argv[1])
    numChannels = int(sys.argv[2])

    # load the whole dataset, for data info
    all_ds   = tfds.load("eurosat", with_info=True)
    # load training, testing & validation sets, splitting by 60%, 20% and 20% respectively
    train_ds = tfds.load("eurosat", split="train[:60%]")
    test_ds  = tfds.load("eurosat", split="train[60%:80%]")
    valid_ds = tfds.load("eurosat", split="train[80%:]")

    num_examples = all_ds[1].splits["train"].num_examples

    batch_size = 64

    # preprocess training & validation sets
    train_ds = prepare_for_training(train_ds, batch_size=batch_size)
    valid_ds = prepare_for_training(valid_ds, batch_size=batch_size)

    model, checkpoint = getModel([None, imageSize, imageSize, numChannels])

    # number of training steps
    n_training_steps   = int(num_examples * 0.6) // batch_size
    # number of validation steps
    n_validation_steps = int(num_examples * 0.2) // batch_size

    history = trainModel(model, epochs, train_ds, valid_ds, n_training_steps, n_validation_steps, checkpoint)