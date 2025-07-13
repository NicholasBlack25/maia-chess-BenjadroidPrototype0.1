import argparse
import os
import yaml
import glob
import numpy as np
import tensorflow as tf


def load_chunk(filepath):
    with np.load(filepath) as data:
        return data['X'], data['Y_policy'], data['Y_value']


def npz_dataset(chunk_paths):
    def generator():
        for path in chunk_paths:
            X, Y_policy, Y_value = load_chunk(path)
            for x, y_p, y_v in zip(X, Y_policy, Y_value):
                print("x:", x.shape, "y_p:", y_p.shape, "y_v:", np.shape(y_v))
                yield x, {
                    'policy': y_p,
                    'value': np.array([y_v], dtype=np.float32)  # wrap scalar y_v to shape (1,)
                }

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32),
            {
                'policy': tf.TensorSpec(shape=(4096,), dtype=tf.float32),
                'value': tf.TensorSpec(shape=(1,), dtype=tf.float32)
            }
        )
    )


def build_model(filters, blocks, value_head_hidden):
    inputs = tf.keras.Input(shape=(8, 8, 12))
    x = inputs

    for _ in range(blocks):
        residual = x
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = tf.keras.layers.Conv2D(filters, 1, padding='same')(residual)

        x = tf.keras.layers.add([x, residual])
        x = tf.keras.layers.ReLU()(x)

    # Policy head
    p = tf.keras.layers.Conv2D(2, 1, activation='relu')(x)
    p = tf.keras.layers.Flatten()(p)
    policy_output = tf.keras.layers.Dense(4096, activation='softmax', name='policy')(p)

    # Value head
    v = tf.keras.layers.Conv2D(1, 1, activation='relu')(x)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(value_head_hidden, activation='relu')(v)
    value_output = tf.keras.layers.Dense(1, activation='tanh', name='value')(v)

    model = tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output])
    return model


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(yaml.dump(config, default_flow_style=False))

    chunk_paths = sorted(glob.glob(config['dataset']['input_train']))
    if len(chunk_paths) == 0:
        print("No .npz chunks found.")
        return

    train_dataset = npz_dataset(chunk_paths)
    train_dataset = train_dataset.shuffle(10000).batch(
        config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)

    model = build_model(
        filters=config['model']['filters'],
        blocks=config['model']['blocks'],
        value_head_hidden=config['model']['value_head_hidden']
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        },
        metrics={
            'policy': 'categorical_accuracy',
            'value': 'mse'
        }
    )

    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=config['training']['num_training_steps']
    )

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{config['name']}.h5")
    print(f"Model saved as models/{config['name']}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
