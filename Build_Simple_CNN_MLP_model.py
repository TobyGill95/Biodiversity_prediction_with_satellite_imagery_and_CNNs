import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg

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

df=pd.read_csv('GBIF_data_output final.csv')

#Variables under analysis
target_variables=['All shannon']

#drop blank rows before subsampling
df=df.dropna(subset=target_variables)

#how many images will be included in the training batch
# batch size limited by small sample when running locally. Try one image at a time
batch_size = 1
#Increase patches per image to full number in full version!
patches_per_image=500
PATCH_SIZE=200
#Increase epochs in full version!
epochs=20
dropout=0.2
#Set number of k-folds for cross validation
fold_num=3

#Add limit to number of training intances, to reduce time spent if needed. 
#Set to None if want to use all
dataset_size_limit=None

#Set file path to file containing jpegs - reset for final version
jpeg_file_path='gbif_sentinel_outputs/jpeg_files/'

#Set whether to save the best model per variable to file
model_save=False

run_locally=True

if run_locally==False:
    #Set file for output log
    log_file = open("Model_outputs/run.log", "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

#Limit data to specified maximum size
if dataset_size_limit is not None:
	df = df.sample(
	n=dataset_size_limit,
	random_state=1
	)
	df=df.reset_index(drop=True)

# Patch extraction
def extract_patches(image):
    patches = tf.image.extract_patches(

        images=tf.expand_dims(image, 0),

        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

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

    del patches

    print(padded.shape)
    return padded

#Create custom layer for counting cnn forward passes
class DebugLayer(layers.Layer):

    def call(self, inputs):
        tf.print(
            "CNN received batch:",
            tf.shape(inputs)
        )
        return inputs


#Function for building the model, recalled inside each fold

def create_model(epochs):
    #Build the cnn 
    cnn = keras.Sequential([

        layers.Input((PATCH_SIZE+6,PATCH_SIZE+6,3)),

        DebugLayer(),

        layers.Conv2D(32,3,activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64,3,activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128,3,activation="relu"),
        #Convert outout to single 128d vector, averaging values across all feature map
        layers.GlobalAveragePooling2D(),

    ])

    cnn.summary()

    #Build the mlp
    mlp = keras.Sequential([

        layers.Input((128,)),

        Dropout(dropout),

        layers.Dense(64,activation="relu"),

        Dropout(dropout),

        layers.Dense(32,activation="relu"),

        layers.Dense(1,activation="linear")
    ])

    mlp.summary()


    #Nesting the models - final model

    inputs = keras.Input((patches_per_image,PATCH_SIZE+6,PATCH_SIZE+6,3))
    #Apply the cnn to all layers to get 128 x patches array
    x = layers.TimeDistributed(cnn)(inputs)
    #average the embeddings 
    x = layers.GlobalAveragePooling1D()(x)
    #feed embeddings into mlp
    outputs = mlp(x)
    #assemble into full model
    full_model = keras.Model(inputs, outputs)

    #Set up training parameters

    opt = 'adam'
    #Mean square error loss for continuous target variable
    loss = 'mse'
    #use mean absolute error (interpretable) and root mean squared error for evaluating performance on unseen data
    metrics = ["mae", tf.keras.metrics.RootMeanSquaredError()]
    #Epochs = how many times to go over whole training set. 
    epochs = epochs
    #Compile the model with training approach
    full_model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return full_model

#Define loading function for pulling tf dataset elements into memory

def load_image(path, label):

    # Read the file from disk
    image = tf.io.read_file(path)


    # Decode the JPEG
    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    #Custom crop function
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

    #Use predefined function to convert jpeg to patches
    patches=extract_patches(cropped)

    #Randomly shuffle the patches (with a random seed) to get random sample from full set
    tf.random.set_seed(1)
    patches = tf.random.shuffle(patches)
    patches = patches[:patches_per_image]

    #Convert to float
    patches = tf.image.convert_image_dtype(
    patches,
    tf.float32
        )
    #rescale
    #patches=patches/255

    del image
    del cropped

    return patches, label

def create_dataset(x,y):
     #Create dataset pipeline 
    dataset = tf.data.Dataset.from_tensor_slices(
        (x, y)
    )

    print('dataset compiled')

    #Apply custom loading function
    dataset = dataset.map(
        load_image,
        #Prevent multiple threads for running locally. Allow autotune on gpu. 
        num_parallel_calls=4
        #num_parallel_calls=tf.data.AUTOTUNE
    )

    print('dataset load functions applied')

    #Define batch size
    dataset = dataset.batch(1)

    #Add prefetching for efficiency
    dataset = dataset.prefetch(
        #No autotune while running locally, prevent multiple threads, add on gpu
        1
        #tf.data.AUTOTUNE
    )

    print('dataset prefetched')

    return dataset


#Set up storage mecahnism for scores
outdf=pd.DataFrame({'Variable':[],'Model type':[],'Epochs':[],'Dropout':[],'Patch size':[],'Patches sampled per image':[],'Loss':[],'MAE':[],'RSME':[]})

#For variables
for target_variable in target_variables:

    print(f'Target variable: {target_variable}')


    #Create benchmark naive model - linear regression 
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


    #Create lists of paths and targets
    image_paths=[]
    targets=[]
    for index, row in df.iterrows():
        tile='T'+row['id']+'.jpg'
        #Check if jpeg image is available for row
        if tile in os.listdir(jpeg_file_path):
            image_paths.append(os.getcwd()+'/'+jpeg_file_path+'/T'+row['id']+'.jpg')
            targets.append(row[target_variable])

    print('Lists created')

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


    #Set up k folds
    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=7)

    #set up system for finding best model out of all the folds
    best_model=None
    best_score=float('inf')
    best_scale_factor=0
    best_epochs=0
    
    fold=1
    #Go through folds
    for train_idx, val_idx in kfold.split(X_full_train):

        print(f'Fold number: {fold}')


        #Split lists into train / test for current fold
        X_train, X_val = X_full_train[train_idx], X_full_train[val_idx]
        y_train, y_val = y_full_train[train_idx], y_full_train[val_idx]

        #Convert targets to tensors
        y_train = tf.constant(y_train, dtype=tf.float32)
        y_val = tf.constant(y_val, dtype=tf.float32)
        

        #Normalise target values (to train max to prevent test leak)
        train_max = tf.reduce_max(y_train)
        y_val = y_val / train_max
        y_train = y_train / train_max

        print('Compiling train and test tensorflow datasets')
       #Compile tensorflow dataset
        train_dataset=create_dataset(X_train,y_train)
        #Create test dataset
        val_dataset=create_dataset(X_val,y_val)

        #Create model
        fold_model=create_model(epochs)

        print('Model created')

        # Set early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                     min_delta=0.001,restore_best_weights=True)

        #Fit model. Save history for tracking epochs before early stopping. 
        history=fold_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        #Implement early stopping from callback set up earlier
        callbacks=[early_stopping]   
        )   

        print('Model trained')
        

        #Score model, record scores for fold
        loss_score, mae, root_mse = fold_model.evaluate(val_dataset, verbose=1)
        print('Model evaluated for fold')


        #Check if this is the best performing model (out of all folds)
        if loss_score<best_score:
            print('Best model so far in fold')
            best_score=loss_score
            best_model=fold_model
            best_epochs=len(history.history['loss'])
            #Capture max of train set so that test set can be normalised using it for testing later
            best_scale_factor=train_max

        #Clear memory
        del history
        del train_dataset
        del val_dataset


        fold=fold+1

    print('All folds completed')

    print('Creating test dataset')
    #Convert to tensor
    y_test = tf.constant(y_test, dtype=tf.float32)
    #Normalise test target values
    y_test = y_test / best_scale_factor
    test_dataset=create_dataset(X_test,y_test)

    print('Evaluating best model against test')
    #Evaluate best model against unseen test data
    loss_score, mae, root_mse = best_model.evaluate(test_dataset, verbose=1)

    print(f'Final scores for variable: Loss: {loss_score}, MAE: {mae}, RMSE: {root_mse}')

    #Record scores and save updated dataframe
    outdf.loc[len(outdf)]=[target_variable,
                            'CNN-MLP',
                               best_epochs,
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
        best_model.save(f'Model_outputs/{target_variable}_model.keras')

    #Clear session for memory
    tf.keras.backend.clear_session()

print('COMPLETE')




    
