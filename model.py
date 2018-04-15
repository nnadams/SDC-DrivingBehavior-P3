import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

def main():
    # Print out all available TensorFlow devices
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()

    for device in devices:
        if device.device_type == 'GPU':
            print(device.physical_device_desc)
        else:
            print(device.name[1:])

    # Read in steering angle from CSV
    lines = []
    with open('recordings/driving_log.csv') as file:
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)

    lines = lines[1:]

    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # Create a batch generator that returns the images and angles
    # Using a generator here avoids storing everything in memory
    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1:
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images, measurements = [], []
                new_images, new_measurements = [], []
                # Process each line from the CSV for the network
                for batch_sample in batch_samples:
                    for i in range(3):
                        path = batch_sample[i]
                        if '\\' in path: # Handle either Windows or UNIX path
                            filename = path.split('\\')[-1]
                        else:
                            filename = path.split('/')[-1]

                        # Center, left and right images are read into
                        # Left and right images have their angles offset
                        location = 'recordings/IMG/' + filename
                        images.append(cv2.imread(location))
                        measurement = float(batch_sample[3])
                        if i == 1:
                            measurement += 0.2
                        elif i == 2:
                            measurement -= 0.2
                        measurements.append(measurement)

                # Each image is duplicated flipped to provide data in both directions
                for image, measurement in zip(images, measurements):
                    new_images.append(image)
                    new_measurements.append(measurement)

                    new_images.append(cv2.flip(image, 1))
                    new_measurements.append(measurement * -1.0)

                # Finally shuffle the data
                X_train = np.array(new_images)
                y_train = np.array(new_measurements)
                yield sklearn.utils.shuffle(X_train, y_train)

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    # Normalize and then crop off top and bottom part of the image
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    # Smaller version of Nvidia network (https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.6))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))
    # No activation is used for regression (ie no softmax)

    # Using Mean Squared Error as loss function
    # "Minimize error between ground truth and predicted measurement"
    model.compile(loss='mse', optimizer='adam')

    # Train for 2 epochs
    # Samples multiplied by 6 (original, original flipped, left, left flipped, right, right flipped)
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=2, verbose=1)

    model.save('model.h5')
    print("Model Saved!")

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
