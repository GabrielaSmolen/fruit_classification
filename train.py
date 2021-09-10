from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import yaml
from tensorflow.keras.models import Sequential


def create_model(config):
    input_shape = (config['img_width'], config['img_height'], config['img_num_channels'])

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(config['no_classes'], activation='softmax'))

    print(model.summary())
    return model


def train(train_datagen, test_datagen, config):
    loss_function = sparse_categorical_crossentropy
    optimizer = Adam()

    model = create_model(config=config)

    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(
        train_datagen,
        epochs=config['no_epochs'],
        validation_data=test_datagen,
        shuffle=True)

    return model


def run(config):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_datagen = train_datagen.flow_from_directory(
        config['train_path'],
        save_to_dir=config['train_dir'],
        save_format='jpeg',
        batch_size=config['batch_size'],
        target_size=(25, 25),
        class_mode='sparse')

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    test_datagen = test_datagen.flow_from_directory(
        config['test_path'],
        save_to_dir=config['test_dir'],
        save_format='jpeg',
        batch_size=config['batch_size'],
        target_size=(25, 25),
        class_mode='sparse')

    model = train(train_datagen, test_datagen, config=config)
    test(config=config, test_datagen=test_datagen, model=model)


def test(config, test_datagen, model):
    result = model.evaluate(test_datagen, verbose=1)
    model.save(config['model_path'])
    print(f"Test loss: {result[0]}")
    print(f"Test acc: {result[1]}")


if __name__ == '__main__':
    with open(os.path.join("config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run(config=config)
