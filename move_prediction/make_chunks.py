import os
import argparse
import numpy as np

def make_chunks(data_path, output_dir, chunk_size=8192):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    X = data['X']
    Y_policy = data['Y_policy']
    Y_value = data['Y_value']

    os.makedirs(output_dir, exist_ok=True)
    total_examples = X.shape[0]
    num_chunks = (total_examples + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_examples)
        chunk = {
            'X': X[start:end],
            'Y_policy': Y_policy[start:end],
            'Y_value': Y_value[start:end],
        }
        chunk_path = os.path.join(output_dir, f'chunk_{i}.npz')
        np.savez_compressed(chunk_path, **chunk)
        print(f"Saved chunk {i} with {end - start} examples to {chunk_path}")

    print(f"Created {num_chunks} chunks in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to training .npz file')
    parser.add_argument('--output-dir', required=True, help='Directory to write chunk files')
    args = parser.parse_args()

    make_chunks(args.data_dir, args.output_dir)
