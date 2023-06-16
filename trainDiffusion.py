# First, train the diffusion model

from src import DiffusionUtils
import tensorflow as tf
import tensorflow_datasets as tfds
import sys


if __name__ == "__main__":
    # Args list: epcohs, img_size, img_channels



    # load training, testing & validation sets, splitting by 60%, 20% and 20% respectively
    (train_ds,) = tfds.load("eurosat", split=["train[:60%]"], with_info=False)
    #(test_ds, ) = tfds.load("eurosat", split=["train[60%:80%]"], with_info=False)
    #(valid_ds,) = tfds.load("eurosat", split={"train[80%:]"}, with_info=False)


    # Get HyperParams
    batch_size = 32
    num_epochs = int(sys.argv[1])  # Just for the sake of demonstration
    total_timesteps = 1000
    norm_groups = 8  # Number of groups used in GroupNormalization layer
    learning_rate = 2e-4

    img_size = int(sys.argv[2])
    img_channels = int(sys.argv[3])
    clip_min = -1.0
    clip_max = 1.0

    first_conv_channels = 64
    channel_multiplier = [1, 2, 4, 8]
    widths = [first_conv_channels * mult for mult in channel_multiplier]
    has_attention = [False, False, True, True]
    num_res_blocks = 2  # Number of residual blocks


    # Build the unet model
    network = DiffusionUtils.build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=tf.keras.activations.swish,
        first_conv_channels=first_conv_channels
    )
    ema_network = DiffusionUtils.build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=tf.keras.activations.swish,
        first_conv_channels=first_conv_channels
    )
    ema_network.set_weights(network.get_weights())  # Initially the weights are the same

    # Get an instance of the Gaussian Diffusion utilities
    gdf_util = DiffusionUtils.GaussianDiffusion(timesteps=total_timesteps)

    # Get the model
    model = DiffusionUtils.DiffusionModel(
        network=network,
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=total_timesteps,
    )

    # Compile the model
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )

    # Train the model
    model.fit(
        train_ds,
        epochs=num_epochs,  
        batch_size=batch_size)

    model.ema_network.save_weights("models/EMA_Saved.h5")

