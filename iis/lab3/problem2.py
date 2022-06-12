import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization,
                                     Dense, Embedding, Flatten, Input, Reshape,
                                     UpSampling1D, multiply)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def plot(sequence, title):
    sns.lineplot(x=range(30), y=sequence)
    plt.title(title)
    plt.savefig('plots/lab3/' + title + '.png')
    plt.show()


def load_data():
    data = pd.read_csv('data/Melbourne_daily_temp.csv')

    sequences = []
    months = []

    for i in range(29, len(data)):
        window = data.loc[i - 29: i]
        last_month = int(window.loc[i]['Date'].split('-')[1])
        sequences.append(window['Temp'].values)
        months.append(last_month)

    scaler = MinMaxScaler()

    sequences = scaler.fit_transform(sequences)
    return np.array(sequences), np.array(months)


def build_discriminator(sequence_shape):
    model = Sequential()

    model.add(LSTM(512, input_dim=np.prod(sequence_shape)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    seq = Input(shape=sequence_shape)
    month = Input(shape=(1,), dtype='int32')
    month_embedding = Flatten()(
        Embedding(12, np.prod(sequence_shape))(month)
    )
    model_input = multiply([seq, month_embedding])

    validity = model(model_input)

    return Model((seq, month), validity)


def build_generator(latent_dim):
    model = Sequential()

    model.add(Dense(15, input_dim=latent_dim))
    model.add(Reshape((15, 1)))

    model.add(UpSampling1D())
    model.add(LSTM(512))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(30, activation='tanh'))

    model.summary()

    noise = Input(shape=(latent_dim,))
    month = Input(shape=(1,), dtype='int32')

    month_embedding = Flatten()(Embedding(12, latent_dim)(month))

    model_input = multiply([noise, month_embedding])
    seq = model(model_input)

    return Model((noise, month), seq)


if __name__ == '__main__':
    X_sequences, X_months = load_data()
    plot(X_sequences[0], f'problem2_real_sequence_month_{X_months[0]}')

    sequence_shape = (30, 1)
    latent_dim = 100
    batch_size = 128
    epochs = 200

    optimizer = Adam(0.0002, 0.5)

    discriminator = build_discriminator(sequence_shape)
    discriminator.compile(loss=['binary_crossentropy'],
                          optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator(latent_dim)

    noise = Input(shape=(latent_dim,))
    month = Input(shape=(1,), dtype='int32')
    seq = generator((noise, month))

    discriminator.trainable = False

    valid = discriminator((seq, month))

    combined = Model((noise, month), valid)
    combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    X_sequences = np.expand_dims(X_sequences, axis=2)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of sequences
        idx = np.random.randint(0, X_sequences.shape[0], batch_size)
        sequences = X_sequences[idx]
        months = X_months[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_months = np.random.randint(0, 12, (batch_size, 1))

        # Generate a half batch of new images
        gen_sequences = generator.predict((noise, gen_months))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch((sequences, months), valid)
        d_loss_fake = discriminator.train_on_batch(
            (gen_sequences, gen_months), fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = combined.train_on_batch((noise, gen_months), valid)

        # Plot the progress
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %
              (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    noise = np.random.normal(0, 1, (1, 100))
    gen_sequence = generator.predict((noise, X_months[[0]]))[0]
    real_sequence = np.reshape(X_sequences[0], (-1))
    plot(gen_sequence, f'problem2_fake_sequence_month_{X_months[0]}')
    real_max_value = np.max(real_sequence)
    fake_max_value = np.max(gen_sequence)
    real_min_value = np.min(real_sequence)
    fake_min_value = np.min(gen_sequence)
    real_mean_value = np.mean(real_sequence)
    fake_mean_value = np.mean(gen_sequence)
    real_range = real_max_value - real_min_value
    fake_range = fake_max_value - fake_min_value
    real_mean_consecutive_diff = np.mean(np.diff(real_sequence))
    fake_mean_consecutive_diff = np.mean(np.diff(gen_sequence))
    print(
        f'Real max value: {real_max_value:.2f},'
        f' fake max value: {fake_max_value:.2f}'
    )
    print(
        f'Real min value: {real_min_value:.2f},'
        f' fake min value: {fake_min_value:.2f}'
    )
    print(
        f'Real mean value: {real_mean_value:.2f},'
        f' fake mean value: {fake_mean_value:.2f}'
    )
    print(
        f'Real range: {real_range:.2f}, fake range: {fake_range:.2f}'
    )
    print(
        f'Real mean consecutive diff: {real_mean_consecutive_diff:.2f},'
        f' fake mean consecutive diff: {fake_mean_consecutive_diff:.2f}'
    )
