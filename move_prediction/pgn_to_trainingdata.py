import chess.pgn
import numpy as np
import sys

def encode_board(board):
    """Convert the board to a 8x8x12 binary matrix (one-hot encoded)."""
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    matrix = np.zeros((8, 8, 12), dtype=np.uint8)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        matrix[row, col, piece_map[piece.symbol()]] = 1
    return matrix

def encode_move(move):
    """Simple encoding of the move as a vector of shape (64*64,)."""
    from_square = move.from_square
    to_square = move.to_square
    index = from_square * 64 + to_square
    policy = np.zeros(64 * 64)
    policy[index] = 1
    return policy

def main(pgn_path, output_path):
    X = []
    Y_policy = []
    Y_value = []

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            result = game.headers.get("Result")
            if result == "1-0":
                game_value = 1
            elif result == "0-1":
                game_value = -1
            else:
                game_value = 0

            board = game.board()
            for move in game.mainline_moves():
                X.append(encode_board(board))
                Y_policy.append(encode_move(move))
                Y_value.append(game_value)
                board.push(move)

    X = np.array(X, dtype=np.uint8)
    Y_policy = np.array(Y_policy, dtype=np.uint8)
    Y_value = np.array(Y_value, dtype=np.int8)

    np.savez_compressed(output_path, X=X, Y_policy=Y_policy, Y_value=Y_value)
    print(f"✓ Saved {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pgn_to_trainingdata.py <input.pgn> <output.npz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
