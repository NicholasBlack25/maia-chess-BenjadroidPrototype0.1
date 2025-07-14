import argparse
import tensorflow as tf
import gzip
from google.protobuf import text_format
import network_format_pb2  # This should be provided by Lc0's protobuf definition

def create_model():
    from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add, Activation
    from tensorflow.keras.models import Model

    input_layer = Input(shape=(8, 8, 112), name='input')
    x = Conv2D(256, 3, padding='same', activation='relu')(input_layer)

    # Residual blocks
    for _ in range(9):
        shortcut = x
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same')(x)
        x = Add()([shortcut, x])
        x = Activation('relu')(x)

    # Policy head
    policy_head = Conv2D(2, 1, activation='relu')(x)
    policy_head = Flatten()(policy_head)
    policy_head = Dense(1858, activation='softmax', name='policy')(policy_head)

    # Value head
    value_head = Conv2D(1, 1, activation='relu')(x)
    value_head = Flatten()(value_head)
    value_head = Dense(256, activation='relu')(value_head)
    value_head = Dense(1, activation='tanh', name='value')(value_head)

    model = Model(inputs=input_layer, outputs=[policy_head, value_head])
    return model

def export_to_lc0_proto(model, output_path):
    import numpy as np
    import network_format_pb2  # This is from Lc0's protobuf, needs to be generated from .proto

    network = network_format_pb2.Net()
    network.version = 1  # Adjust as per Lc0 expected version

    weights = model.get_weights()
    for w in weights:
        network.weights.append(w.flatten().tolist())

    with gzip.open(output_path, 'wb') as f:
        f.write(network.SerializeToString())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a trained Maia model to Lc0 .pb.gz format.')
    parser.add_argument('--weights', required=True, help='Path to the Maia Keras .h5 model')
    parser.add_argument('--output', required=True, help='Output path for the .pb.gz file')
    args = parser.parse_args()

    print(f"Loading Maia model weights from {args.weights}...")
    model = create_model()
    model.load_weights(args.weights)
    print("Weights loaded.")

    print(f"Exporting to Lc0 protobuf format at {args.output}...")
    export_to_lc0_proto(model, args.output)
    print("Export complete.")
