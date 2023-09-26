import numpy as np
from itertools import chain
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def _indexing(x, indices):
    """
    :param x: array from which indices has to be fetched
    :param indices: indices to be fetched
    :return: sub-array from given array and indices
    """
    # np array indexing
    if hasattr(x, 'shape'):
        return x[indices]

    # list indexing
    return [x[idx] for idx in indices]

def train_test_split(*arrays, test_size=0.25, shuffle=True, random_seed=1):
    """
    splits array into train and test data.
    :param arrays: arrays to split in train and test
    :param test_size: size of test set in range (0,1)
    :param shufffle: whether to shuffle arrays or not
    :param random_seed: random seed value
    :return: return 2*len(arrays) divided into train ans test
    """
    # checks
    assert 0 < test_size < 1
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length*test_size))
    n_train = length - n_test

    if shuffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))

def train_model():
  # Loading the data
  data_path = 'C:\\Users\\krond\\workspace\\wooly-mask-detector\\wooly_mask_detector\\dataset'
  categories = os.listdir(data_path)
  labels = [i for i in range(len(categories))]
  label_dict = dict(zip(categories, labels))

  # Data preprocessing
  img_size = (160, 160) #image size 
  data = []
  target = []

  for category in categories: #without_mask and with_mask
    folder_path = os.path.join(data_path, category) #Getting the subfolder path
    img_names = os.listdir(folder_path) #Listing all items in that subfolder

    for img_name in img_names: #All the images in that subfolder
      img_path = os.path.join(folder_path, img_name) #Getting the image path
      img = cv2.imread(img_path) #Reading the image 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converting from BGR to RGB

      try:
        resized = cv2.resize(img, img_size) #Image resizing

        data.append(resized)
        target.append(label_dict[category])

      except Exception as e:
        print("Exception: ", e)
      
  data = np.array(data)/255.0 #Rescaling
  data = np.reshape(data, (data.shape[0], img_size[0], img_size[1], 3)) #Reshaping 
  target = np.array(target)
  target = tf.keras.utils.to_categorical(target)

  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, shuffle = True)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, shuffle = True)


  datagen = tf.keras.preprocessing.image.ImageDataGenerator( # Data Augmentation
          rotation_range=15,
          zoom_range = 0.15,  
          width_shift_range=0.1, 
          height_shift_range=0.1)

  # VGG16 transfer learning
  vgg = tf.keras.applications.VGG16(input_shape = (160, 160, 3), weights = 'imagenet', include_top = False)

  for layer in vgg.layers: # Setting all VGG16 layers false for training.
    layer.trainable = False

  print(vgg.output)
  x = tf.keras.layers.Flatten()(vgg.output)

  prediction = tf.keras.layers.Dense(units = 2, activation = 'softmax')(tf.keras.layers.Dense(units = 64, activation = 'relu')(x)) # Adding dense layer

  model = tf.keras.models.Model(inputs = vgg.input, outputs = prediction) # Joining the pre-training convolutional layers and dense layers

  print(model.summary())

  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  history = model.fit(datagen.flow(X_train, y_train, batch_size=32),  # Training the model.
                epochs=11, validation_data = (X_val, y_val))


  # Model accuracy
  correct = 0
  total = 0
  pred = np.argmax(model.predict(X_test), axis=1)

  for i, img in enumerate(pred):
    if img == np.argmax(y_test[i]):
      correct += 1
    total += 1

  print(correct/total * 100)

  # Plotting the model losses and accuracies
  plt.plot(np.arange(0, 11), history.history['val_loss'], label='val_loss')
  plt.plot(np.arange(0, 11), history.history['loss'], label = 'loss')
  plt.plot(np.arange(0, 11), history.history['accuracy'], label = 'accuracy')
  plt.plot(np.arange(0, 11), history.history['val_accuracy'], label = 'val_accuracy')
  plt.legend()
  plt.show()

  # Saving the model
  model.save("wooly_mask_detector/models/maskclassifier.model", save_format = "h5")

