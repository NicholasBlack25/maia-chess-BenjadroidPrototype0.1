import argparse
import os
import chess.pgn
import numpy as np
from trainingdata_tool import generate_training_data

def get_all_pgn_paths(input_path):
    if os.path.isdir(input_path):
        return [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".pgn")]
    else:
        return [input_path]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to PGN file or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output .npz file')
    args = parser.parse_args()

    pgn_paths = get_all_pgn_paths(args.input)

    all_data = []
    for path in pgn_paths:
        with open(path, encoding="utf-8") as pgn_file:
            print(f"Processing {path}...")
            data = generate_training_data(pgn_file)
            all_data.extend(data)

    print(f"Saving {len(all_data)} positions to {args.output}...")
    np.savez_compressed(args.output, positions=all_data)

if __name__ == '__main__':
    main()
