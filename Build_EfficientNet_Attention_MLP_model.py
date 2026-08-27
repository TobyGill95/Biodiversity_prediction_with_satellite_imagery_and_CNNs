import keras
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg

import json

import pandas as pd

import numpy as np
import os

from PIL import Image, ImageOps

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#Load pretrained model
from tensorflow.keras.applications import EfficientNetB0

run_locally=False
#how many images will be included in the training batch
# batch size limited by small sample when running locally. Try one image at a time
batch_size = 1
#Increase patches per image to full number in full version!
patches_per_image=500
PATCH_SIZE=200
#Increase epochs in full version!
epochs=20
dropout=0.2


#Add limit to number of training intances, to reduce time spent if needed. 
#Set to None if want to use all
dataset_size_limit=None

#Set file path to file containing jpegs - reset for final version
jpeg_file_path='jpeg_files/'

#Set whether to save the best model per variable to file
model_save=True


df=pd.read_csv('GBIF_data_output final.csv')


#Variables under analysis
target_variables=['All shannon']

#Filter dataframe to only the images available locally
keeps=[]
for tile in df['id']:
    if 'T'+tile+'.jpg' in os.listdir('jpeg_files'):
        keeps.append(True)
    else:
        keeps.append(False)
df=df[keeps]
print(len(df))
    

#Reduce dataframe to chosen sample size
df = df.dropna(subset=target_variables)
#Limit data to specified maximum size
if dataset_size_limit is not None:
    df = df.sample(
    n=dataset_size_limit,
    random_state=1
    )
    df=df.reset_index(drop=True)
    print(len(df))
    print(df.head())

#Set instructions for where to print outputs if running on slurm
if run_locally==False:
    #Set file for output log
    log_file = open("Model_outputs/run.log", "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file


# Patch extraction
def extract_patches(image):

    #Use tensorflow patch extraction function
    patches = tf.image.extract_patches(

        images=tf.expand_dims(image, 0),

        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    #reshape for batch size
    patches = tf.reshape(
        patches,
        (-1, PATCH_SIZE, PATCH_SIZE, 3)
        )

    #Add padding
    padded = tf.pad(
        patches,
        #no padding on batch, pad 3 on height and width each side, none for channels
        paddings=[
            [0, 0],  
            [3, 3],  
            [3, 3],  
            [0, 0]   
        ],
        mode="CONSTANT",
        constant_values=0
    )

    #delete variable to save memory
    del patches

    print(padded.shape)
    return padded



#Define loading function for pulling tf dataset elements into memory

def load_image(path, label):

    # Read the file from disk
    image = tf.io.read_file(path)


    # Decode the JPEG
    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    #Custom crop function to get consistent 10k x 10k
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    offset_y = (height - 10000) // 2
    offset_x = (width - 10000) // 2
    cropped = image[
        offset_y:offset_y + 10000,
        offset_x:offset_x + 10000,
        :
    ]

    # Extract patches
    patches = extract_patches(cropped)

    # NOTE - LINE ASSUMES 200 X 200 PATCHES
    patches.set_shape([2500, 206, 206, 3])

    #Create list of patches to keep
    selected_patches = []

    #Create function for filtering out the blank/empty/uninformative patches
    #Process in chunks to save memory, due to jpeg file size
    print('Checking quality of patches')
    # Number of patches to process simultaneously
    chunk_size = 64

    selected_patches = []

    num_patches = patches.shape[0]

    #Iterate through patches in vectorised way for efficiency
    for start in range(0, num_patches, chunk_size):

        end = min(start + chunk_size, num_patches)

        # Small chunk of patches
        patch_chunk = patches[start:end]

        # Convert only this chunk to float (memory limitations)
        patch_chunk = tf.image.convert_image_dtype(
            patch_chunk,
            tf.float32
        )

        # Compute std for the whole chunk in parallel
        std = tf.math.reduce_std(
            patch_chunk,
            axis=[1, 2, 3]
        )

        # Keep informative patches (std of 0.11 found to be best threshold for removing empty patches)
        keep = std > 0.11

        #Add to list of informative patches, in vectorised way
        selected_patches.append(
            tf.boolean_mask(patch_chunk, keep)
        )

    # Combine selected patches
    patches = tf.concat(selected_patches, axis=0)

    #RETURN IMAGES TO 0-255 INT SCALING FOR EFFICIENTNET INPUT
    patches =tf.cast(tf.round(patches * 255.0), tf.uint8)

    # Are there sufficient informative patches in the image to meet the required sample size? 
    enough_patches = tf.shape(patches)[0] >= patches_per_image

    if enough_patches:

        #Get indices for slicing correct patches
        indices = tf.random.shuffle(
            tf.range(tf.shape(patches)[0])
            #MADE PATCH SAMPLING DETERMINISTIC FOR LOCAL MODEL TESTING
            ,seed=1,
        )[:patches_per_image]
        #Keep only the correct patches
        patches = tf.gather(patches, indices)

    else:

        patches = tf.zeros(
            (0, 206, 206, 3),
            dtype=tf.uint8
        )

        print("Image discarded due to insufficient patches")

    del image
    del cropped

    print("Patch tensor shape:", patches.shape)

    return patches, label, enough_patches

def create_dataset(x,y):
     #Create dataset pipeline 
    dataset = tf.data.Dataset.from_tensor_slices(
        (x, y)
    )

    print('dataset compiled')

    #Apply custom image loading function
    dataset = dataset.map(
        load_image,
        num_parallel_calls=4
    )

    # Remove images with insufficient patches, using boolean flag returned from patch extraction function
    dataset = dataset.filter(
        lambda patches, label, enough_patches: enough_patches
    )

    # Remove the boolean flag for enough patches produced by the loading function as no longer needed
    dataset = dataset.map(
        lambda patches, label, enough_patches: (patches, label)
    )


    print('dataset load functions applied')

    #Define batch size
    dataset = dataset.batch(batch_size)

    #Add prefetching for efficiency, matching batch size of 1 for memory saving
    dataset = dataset.prefetch(
        1
    )

    print('dataset prefetched')

    return dataset


#Define the model architecture
def create_model(epochs):

    #Build the cnn - use efficientnet pretrained model, the smallest one
    cnn = EfficientNetB0(
    #Exclude softmax classification layer
    include_top=False,
    weights="imagenet",
    #Pool output
    pooling="avg",
    input_shape=(PATCH_SIZE+6, PATCH_SIZE+6, 3)
    )

    #Set cnn so weights are not updated during training, only defaults are used - way faster (as model is many times larger)
    #Consider saving embeddings to disk, so cnn can be cut out of model entirely after one set of forward passes. 
    cnn.trainable = False
    #Set top 30% of layers to be trainable
    n_layers = len(cnn.layers)
    n_unfreeze = int(n_layers * 0.3)
    for layer in cnn.layers[-n_unfreeze:]:
        # Keep BatchNorm frozen
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


    cnn.summary()


    #Build the mlp
    mlp = keras.Sequential([

        layers.Input((1280,)),

        Dropout(dropout),

        layers.Dense(256,activation="relu"),

        Dropout(dropout),

        layers.Dense(64,activation="relu"),

        layers.Dense(1,activation="linear")
    ])

    mlp.summary()


    #Nesting the models - final model

    inputs = keras.Input((patches_per_image,PATCH_SIZE+6,PATCH_SIZE+6,3))
    #Apply the cnn to all layers to get 128 x patches array
    x = layers.TimeDistributed(cnn)(inputs)

    
    #Create attention network for determining key patches
    #reduce to 128 vecetor
    attention = layers.Dense(128, activation="tanh")(x)
    #reduce to a single value per patch
    attention = layers.Dense(1)(attention)
    #Softmax to scale to sum to 1
    attention = layers.Softmax(axis=1)(attention)

    #Add attention layer, multiply patch values by weights
    x = layers.Multiply()([x, attention])
    #Output is dimension batch, patches, 1280, but values scaled

    #Reduce dimensions to single vector by summing across patches (Not mean, as softmax already makes proportionate)
    #Output is dimension batch, 1280
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    #feed embeddings into mlp
    outputs = mlp(x)

    #assemble into full model
    full_model = keras.Model(inputs, outputs)

    #Set up training parameters

    #Adam optimiser with a reduced learning rate
    opt =keras.optimizers.Adam(learning_rate=1e-5) 
    #Mean square error loss for continuous target variable
    loss = 'mse'
    #use mean absolute error (interpretable) and root mean squared error for evaluating performance on unseen data
    metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]
    #Epochs = how many times to go over whole training set. 
    epochs = epochs

    #Compile the model with training approach
    full_model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return full_model


#Begin full loop

#Set up storage mecahnism for scores
outdf=pd.DataFrame({'Variable':[],'Model type':[],'Epochs':[],'Dropout':[],'Patch size':[],'Patches sampled per image':[],'Loss':[],'MAE':[],'RSME':[]})


#For variables
for target_variable in target_variables:

    print(f'Target variable: {target_variable}')


    #Create benchmark model - linear regression 
    print('Creating linear vegetation model')
    dropdf=df[~pd.isna(df['vegetation'])]
    dropdf=dropdf[~pd.isna(dropdf[target_variable])]
    lin_x_train, lin_x_test, lin_y_train, lin_y_test = train_test_split(
    np.array(dropdf['vegetation']).reshape(-1, 1),
    np.array(dropdf[target_variable]),
    test_size=0.2,
    random_state=1
    )
    lin_scaler=np.max(lin_y_train) 
    lin_y_train=lin_y_train/lin_scaler
    lin_y_test=lin_y_test/lin_scaler
    linreg=LinearRegression().fit(lin_x_train,lin_y_train)
    lin_y_pred=linreg.predict(lin_x_test)
    lin_rmse = np.sqrt(mean_squared_error(lin_y_test, lin_y_pred))
    print("Linear vegetation model Root Mean Square Error (RMSE):", lin_rmse)
    outdf.loc[len(outdf)]=[
        target_variable,
        'Linear regression on vegetation',
        #epochs
        np.nan,
        #dropout
        np.nan,
        #patch size
        np.nan,
        #patch sample
        np.nan,
        #loss
        np.nan,
        #mse
        np.nan,
        lin_rmse

    ]
    

    #Create lists of paths and targets for dataset creation
    image_paths=[]
    targets=[]
    for index, row in df.iterrows():
        tile='T'+row['id']+'.jpg'
        #Check if jpeg image is available for row
        if tile in os.listdir(jpeg_file_path):
            image_paths.append(os.getcwd()+'/'+jpeg_file_path+'/T'+row['id']+'.jpg')
            targets.append(row[target_variable])

    print('Lists created')
    print(len(image_paths))

    image_paths=np.array(image_paths)
    targets=np.array(targets)

    #Set up train test split
    #Full train is for model develoment, gets split into train and val in each fold
    #Test is for model eval
    X_full_train, X_test, y_full_train, y_test = train_test_split(
    image_paths,
    targets,
    test_size=0.2,
    random_state=1
    )
    print('Train and test split complete')


    #Split train into pure train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train,
        y_full_train,
        test_size=0.2,
        random_state=1
        )
    print('Train and validation split complete')


    #Convert targets to tensors
    y_train = tf.constant(y_train, dtype=tf.float32)
    y_val = tf.constant(y_val, dtype=tf.float32)

    print('Converted to tensors')
    

    #Normalise target values (to train max to prevent test leak)
    train_max = tf.reduce_max(y_train)
    y_val = y_val / train_max
    y_train = y_train / train_max

    print('Target values normalised')

    print('Compiling train and test tensorflow datasets')
    #Compile tensorflow dataset
    train_dataset=create_dataset(X_train,y_train)
    #Create test dataset
    val_dataset=create_dataset(X_val,y_val)

    #Create model
    model=create_model(epochs)

    model.save('test_model.keras')

    print('Model created')

    # Set early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                    min_delta=0.001,restore_best_weights=True)


    #Fit model. Save history for tracking epochs before early stopping. 
    history=model.fit(
    train_dataset, 
    validation_data=None,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    #Implement early stopping from callback set up earlier
    callbacks=[early_stopping]   
    )   


    with open(f"Model_outputs/training_history_{target_variable}.json", "w") as f:
        json.dump(history.history, f)

    print('Model trained')


    print('Creating test dataset')
    #Convert to tensor
    y_test = tf.constant(y_test, dtype=tf.float32)
    #Normalise test target values
    y_test = y_test / train_max
    test_dataset=create_dataset(X_test,y_test)

    print('Evaluating  model against test')
    #Evaluate best model against unseen test data
    loss_score, mae, root_mse = model.evaluate(test_dataset, verbose=1)

    print(f'Final scores for variable: Loss: {loss_score}, MAE: {mae}, RMSE: {root_mse}')

    #Record scores and save updated dataframe
    outdf.loc[len(outdf)]=[target_variable,
                            'EfficientNet-Attention-MLP',
                                len(history.epoch),
                                dropout,
                                PATCH_SIZE,
                                patches_per_image,
                                loss_score,
                                mae,
                                root_mse]


    outdf.to_csv('Model_outputs/Model_evaluation.csv', index=False)
    print('Output file updated')

    #Option to save best model
    if model_save==True:
        model.save_weights(
        f"Model_outputs/model_{target_variable}.weights.h5")

    #Clear memory
    del history
    del train_dataset
    del val_datasets

    #Clear session for memory
    tf.keras.backend.clear_session()

print('COMPLETE')




