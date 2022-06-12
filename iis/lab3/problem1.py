import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout,
                                     Embedding, Flatten, Input, LeakyReLU,
                                     MaxPooling2D, Reshape, ZeroPadding2D,
                                     multiply)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical


def build_generator(latent_dim, img_shape, num_classes):
    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(14 * 14 * 32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((14, 14, 32)))
    model.add(Conv2DTranspose(
        1, kernel_size=3, strides=2, padding='same', activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)


def build_discriminator(img_shape, num_classes):
    model = Sequential()

    model.add(Reshape(img_shape, input_dim=np.prod(img_shape)))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(
        Embedding(num_classes, np.prod(img_shape))(label)
    )
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)


def build_classifier(img_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', input_shape=img_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def sample_images(generator):
    r, c = 2, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.arange(0, 10).reshape(-1, 1)

    gen_imgs = generator.predict([noise, sampled_labels])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].set_title('Digit: %d' % sampled_labels[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig('plots/lab3/problem1.png')
    plt.show()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # plot_image(X_train[0], y_train[0])

    img_shape = (28, 28, 1)
    num_classes = 10
    latent_dim = 100
    batch_size = 128
    epochs = 1000

    optimizer = Adam(0.0002, 0.5)

    discriminator = build_discriminator(img_shape, num_classes)
    discriminator.compile(loss=['binary_crossentropy'],
                          optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator(latent_dim, img_shape, num_classes)

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([noise, label])

    discriminator.trainable = False

    valid = discriminator([img, label])

    combined = Model([noise, label], valid)
    combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate a half batch of new images
        gen_imgs = generator.predict([noise, labels])

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Condition on labels
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

        # Train the generator
        g_loss = combined.train_on_batch([noise, sampled_labels], valid)

        # Plot the progress
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %
              (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    sample_images(generator)

    mnist_classifier = build_classifier(img_shape, num_classes)
    mnist_classifier.fit(
        X_train, to_categorical(y_train, 10), epochs=30,
        batch_size=batch_size, validation_split=0.2
    )

    X_eval = []
    y_eval = []
    for digit in range(10):
        for i in range(100):
            noise = np.random.normal(0, 1, (1, 100))
            label = np.array([digit])
            gen_img = generator.predict([noise, label])[0]
            X_eval.append(gen_img)
            y_eval.append(digit)
    accuracy_test = mnist_classifier.evaluate(
        X_test, to_categorical(y_test, 10), batch_size=batch_size
    )
    accuracy_gen = mnist_classifier.evaluate(
        np.array(X_eval), to_categorical(np.array(y_eval), 10),
        batch_size=batch_size
    )
    print(f'Accuracy on test set: {accuracy_test[1] * 100:.2f}%')
    print(f'Accuracy on generated set: {accuracy_gen[1] * 100:.2f}%')
