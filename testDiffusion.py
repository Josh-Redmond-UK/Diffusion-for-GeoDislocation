import DiffusionUtils 
import RSIRUtils 
import tensorflow as tf
import tensorflow_datasets as tfds


rsirModel = RSIRUtils.getModel()
rsirModel.load_weights('models/satellite-classification.h5')

featureExtraction = tf.keras.Model(inputs=rsirModel.input,
                                 outputs=rsirModel.layers[0].output)



# Get HyperParams for Diffusion Model
batch_size = 32
num_epochs = 1  # Just for the sake of demonstration
total_timesteps = 1000
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 64
img_channels = 3
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
)
ema_network = DiffusionUtils.build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=tf.keras.activations.swish,
)
ema_network.set_weights(network.get_weights())  # Initially the weights are the same
ema_network.load_weights("models/EMA_Saved.h5")

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


###### Load data
train_ds = tfds.load("eurosat", split="train[:60%]")
test_ds  = tfds.load("eurosat", split="train[60%:80%]")
valid_ds = tfds.load("eurosat", split="train[80%:]")


### Test diffusion
### Get the valid DS 
# Generate the features and classifications
# Do lookup (estimate weighted distance) for each image against the whole dataset
# Then do the same with different diffusion anonymisations
# Estimate - two things
# The average distance from the matching image
# Whether or not the matching image was in the top n 
# Average nearest N for true image

# First, generate image features and classifications



poiNums = [4,6,8]
bufferRanges = [(8,12), (14,16), (18,22)]

listOfArrays = []
rIter = tfds.as_numpy(valid_ds)

for t in rIter:
  listOfArrays.append(t['image'])

rSet = np.stack(listOfArrays)

rClasses = rsirModel.predict(rSet)
rFeatures = featureExtraction.predict(rSet)


# Now estimate the distances to all the images for different diffused models
wResults = []
nResults = [] 
params = []

for pn in poiNums:
    for br in bufferRanges:
        tempDs = valid_ds.map(lambda x: DiffusionUtils.create_poi_image(np.squeeze(x), pn, br))
        result= tempDs.map(lambda x: DiffusionUtils.diffuseRepaint(x, model))

        listOfArrays = []
        qIter = tfds.as_numpy(valid_ds)

        for t in qIter:
            listOfArrays.append(t['image'])

        qSet = np.stack(listOfArrays)

        weights = []
        topNs = []

        for idx, q in enumerate(qSet):
            imgIdxs, weights = RSIRUtils.getNearestIdxs(q, rClasses, rFeatures, rsirModel, featureExtraction, len(qSet), True)
            wd = weights[idx]
            n, = np.where(imgIdxs == idx) # integers
            weights.append(wd)
            topNs.append(n)

        wMean = np.mean(weights)
        nMean = np.mean(topNs)
        params.append([pn, br])
        wResults.append(wMean)
        nResults.append(nMean)



