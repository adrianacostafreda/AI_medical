# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set()

import tensorflow

# Import data generator from keras
from keras.preprocessing.image import ImageDataGenerator

# There are two dataset PruneCXR and LongTailCXR

# Read csv file containing training datadata - PruneCXR
train_df = pd.read_csv("/Users/adriana/Documents/GitHub/AI_medical/nih_prune_train_dataset.csv")
# Print first 5 rows
print(f'There are {train_df.shape[0]} rows and {train_df.shape[1]} columns in this data frame')
print(train_df.head())

# Look at the data type of each column and whether null values are present
print(train_df.info())

# train_df[].value_counts() identifies the unique labels of the subject
print(f"The total patient ids are {train_df['subj_id'].count()}, from those the unique ids are {train_df['subj_id'].value_counts().shape[0]} ")

# As you can see, the number of unique patients in the dataset is less than the total number so there must be some overlap. 
# For patients with multiple records, you'll want to make sure they do not show up in both training and test sets in order 
# to avoid data leakage (covered later in this week's lectures).

columns = train_df.keys()
columns = list(columns)
print(columns)

# Remove unnecesary elements
columns.remove('id')
columns.remove('subj_id')
# Get the total classes
print(f"There are {len(columns)} columns of labels for these conditions: {columns}")

# Random images - we cannot run it since there are images which have not been downloaded

# Print out the number of positive labels for each class
for column in columns:
    print(f"The class {column} has {train_df[column].sum()} samples")

# Extract numpy values from Image column in data frame
images = train_df['id'].values

# Extract 9 random images from it
random_images = images[0:9]

# Location of the image dir
img_dir = '/Users/adriana/Documents/GitHub/AI_medical/images/images/'

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(20,10))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()
plt.show()   

# Get the first image that was listed in the train_df dataframe
sample_img = train_df.id[0]
raw_image = plt.imread(os.path.join(img_dir, sample_img))
plt.imshow(raw_image, cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")

# Plot a histogram of the distribution of the pixels
sns.displot(raw_image.ravel(), 
             label=f'Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')

plt.tight_layout()
plt.show()

# Normalize images
image_generator = ImageDataGenerator(
    samplewise_center=True, #Set each sample mean to 0.
    samplewise_std_normalization= True # Divide each input by its standard deviation
)
