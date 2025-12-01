import os
import json
import pandas as pd
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional

# Load the chess games CSV
def load_chess_dataset(csv_path: str, min_elo: int = 2000) -> pd.DataFrame:

    print("=" * 60)
    print("PART 1: Loading Chess Dataset")
    print("=" * 60)

    # Load the CSV file
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    original_count = len(df)
    print(f"Total games loaded: {original_count:,}")

    # Check if ELO columns exist (full dataset vs pre-filtered)
    has_elo = 'WhiteElo' in df.columns and 'BlackElo' in df.columns

    if has_elo:
        # Full dataset - filter by ELO
        print(f"\nFiltering games where WhiteElo >= {min_elo} AND BlackElo >= {min_elo}...")
        df_filtered = df[(df['WhiteElo'] >= min_elo) & (df['BlackElo'] >= min_elo)].copy()
        print(f"Games after ELO filter: {len(df_filtered):,}")
    else:
        # Pre-filtered dataset - only AN column, skip ELO filtering
        print("\nDataset already pre-filtered (no ELO columns found)")
        df_filtered = df.copy()

    if 'AN' not in df_filtered.columns:
        raise ValueError("AN column not found in CSV")

    # Drop rows where AN is empty or NaN
    df_filtered = df_filtered.dropna(subset=['AN'])
    df_filtered = df_filtered[df_filtered['AN'].str.strip() != '']

    print(f"Games after dropping empty AN: {len(df_filtered):,}")

    # Summary
    print("\n--- Summary ---")
    print(f"Original games: {original_count:,}")
    print(f"Valid games: {len(df_filtered):,}")
    print(f"Removed: {original_count - len(df_filtered):,} games")

    return df_filtered.reset_index(drop=True)

 # Parse the AN string into a list of SAN moves.
def parse_san_moves(an_string: str) -> List[str]:

    # Remove move numbers and split
    tokens = an_string.split()
    moves = []
    for token in tokens:
        # Skip move numbers (e.g., "1.", "2.", etc.)
        if token.endswith('.'):
            continue
        # Skip result indicators
        if token in ['1-0', '0-1', '1/2-1/2', '*']:
            continue
        moves.append(token)
    return moves


def extract_training_pairs(df: pd.DataFrame, max_games: Optional[int] = None) -> pd.DataFrame:
    """
    Extract (FEN, move_san, move_uci) training pairs from games.

    Returns DataFrame with columns: ['fen', 'move_san', 'move_uci']
    """
    print("\n" + "=" * 60)
    print("PART 2: Converting Games to Training Examples")
    print("=" * 60)

    training_data = []

    games_to_process = df if max_games is None else df.head(max_games)

    skipped_moves = 0
    total_moves = 0

    for idx, row in tqdm(games_to_process.iterrows(), total=len(games_to_process), desc="Processing games"):
        an_string = row['AN']
        san_moves = parse_san_moves(an_string)

        # Create a new board for each game
        board = chess.Board()

        for san_move in san_moves:
            total_moves += 1
            fen_before = board.fen()

            try:
                # Parse the SAN move
                move = board.parse_san(san_move)
                uci_move = move.uci()

                # Store the training pair
                training_data.append({
                    'fen': fen_before,
                    'move_san': san_move,
                    'move_uci': uci_move
                })

                # Apply the move to advance the board
                board.push(move)

            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError) as e:
                # Skip illegal or unparseable moves
                skipped_moves += 1
                break  # Game is corrupted from this point, skip remaining moves

    print(f"\nTotal moves processed: {total_moves:,}")
    print(f"Skipped moves (illegal/unparseable): {skipped_moves:,}")
    print(f"Valid training pairs: {len(training_data):,}")

    return pd.DataFrame(training_data)


# Part 3: Encode FEN as Tensor & Move Encoding

# Piece to plane index mapping
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,   # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
}

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string to a tensor with shape (17, 8, 8).

    Planes:
        0-5:   White pieces (P, N, B, R, Q, K)
        6-11:  Black pieces (p, n, b, r, q, k)
        12:    Side to move (1 if white, 0 if black)
        13:    White kingside castling
        14:    White queenside castling
        15:    Black kingside castling
        16:    Black queenside castling
        (En passant encoded implicitly in piece positions)
    """
    tensor = torch.zeros(17, 8, 8, dtype=torch.float32)

    parts = fen.split(' ')
    board_str = parts[0]
    side_to_move = parts[1]
    castling = parts[2] if len(parts) > 2 else '-'

    # Parse piece positions
    row = 7  # FEN starts from rank 8 (index 7)
    col = 0

    for char in board_str:
        if char == '/':
            row -= 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            if char in PIECE_TO_PLANE:
                plane = PIECE_TO_PLANE[char]
                tensor[plane, row, col] = 1.0
            col += 1

    # Side to move plane
    if side_to_move == 'w':
        tensor[12, :, :] = 1.0

    # Castling rights
    if 'K' in castling:
        tensor[13, :, :] = 1.0
    if 'Q' in castling:
        tensor[14, :, :] = 1.0
    if 'k' in castling:
        tensor[15, :, :] = 1.0
    if 'q' in castling:
        tensor[16, :, :] = 1.0

    return tensor


def create_move_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create UCI move to class index mapping

    Uses a 4096-class encoding: 64 from-squares * 64 to-squares
    Also handles promotions by adding extra classes

    Returns tuple of (uci_to_idx, idx_to_uci) dictionaries
    """
    uci_to_idx = {}
    idx_to_uci = {}

    squares = [chess.square_name(i) for i in range(64)]

    # Basic moves: from_square × to_square = 64 × 64 = 4096
    idx = 0
    for from_sq in squares:
        for to_sq in squares:
            uci = from_sq + to_sq
            uci_to_idx[uci] = idx
            idx_to_uci[idx] = uci
            idx += 1

    # Promotion moves (add q, r, b, n suffixes)
    # Additional classes beyond 4096
    promotion_pieces = ['q', 'r', 'b', 'n']
    for from_sq in squares:
        for to_sq in squares:
            for promo in promotion_pieces:
                uci = from_sq + to_sq + promo
                uci_to_idx[uci] = idx
                idx_to_uci[idx] = uci
                idx += 1

    return uci_to_idx, idx_to_uci


def uci_to_class_index(uci_move: str, uci_to_idx: Dict[str, int]) -> int:
    # Convert a UCI move string to a class index.
    return uci_to_idx.get(uci_move, -1)


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess position-move pairs.
    """

    def __init__(self, df: pd.DataFrame, uci_to_idx: Dict[str, int]):
        self.fens = df['fen'].tolist()
        self.moves = df['move_uci'].tolist()
        self.uci_to_idx = uci_to_idx

        # Pre-filter valid moves
        self.valid_indices = []
        for i, move in enumerate(self.moves):
            if move in uci_to_idx:
                self.valid_indices.append(i)

        print(f"Dataset: {len(self.valid_indices):,} valid samples out of {len(self.fens):,}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        actual_idx = self.valid_indices[idx]
        fen = self.fens[actual_idx]
        move = self.moves[actual_idx]

        # Convert FEN to tensor
        tensor = fen_to_tensor(fen)

        # Convert move to class index
        label = self.uci_to_idx[move]

        return tensor, label

class ChessCNN(nn.Module):
    """
    CNN for chess move prediction

    Architecture:
        - Multiple conv layers with batch normalization and ReLU
        - Global average pooling
        - Fully connected layers
        - Output: 4096 + promotion classes
    """

    def __init__(self, num_classes: int = 4096 + 64*64*4):
        super(ChessCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First block: 17 -> 64 channels
            nn.Conv2d(17, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Third block: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Fourth block: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Fifth block: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 0.001
) -> nn.Module:
    # Train the chess CNN model
    
    print("\n" + "=" * 60)
    print("PART 4: Training CNN Model")
    print("=" * 60)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    print(f"\nDevice: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_x, batch_y in train_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch_x, batch_y in val_pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # Update lr
        # TO-DO: Check if this is necessary when using Adam optimizer
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model

def save_model(
    model: nn.Module,
    uci_to_idx: Dict[str, int],
    idx_to_uci: Dict[int, str],
    model_path: str = "model_v2.pth",
    mapping_path: str = "move_mappings.json"
):
    # Save the trained model and move mappings.

    print("\n" + "=" * 60)
    print("PART 5: Saving Model and Mappings")
    print("=" * 60)

    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # Save move mappings
    # Convert int keys to strings
    mappings = {
        'uci_to_idx': uci_to_idx,
        'idx_to_uci': {str(k): v for k, v in idx_to_uci.items()}
    }

    with open(mapping_path, 'w') as f:
        json.dump(mappings, f)

    print(f"Move mappings saved to: {mapping_path}")
    print(f"Total move classes: {len(uci_to_idx):,}")


def main():
    # Main training pipeline - runs the entire thing
    
    # Configuration
    CSV_PATH = "elite_games.csv"  # Path to the chess dataset
    MIN_ELO = 2000 # To filter out lower rated games
    MAX_GAMES = None  # Set to a number to limit games for testing (e.g., 1000)
    BATCH_SIZE = 1024
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.9

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    #elif torch.backends.mps.is_available():
    #   device = torch.device('mps') # for training on MacOS; NOT TESTED
    else:
        device = torch.device('cpu')

    print("=" * 60)
    print("CHESS MOVE PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nDevice: {device}")

    # Part 1: Load dataset
    df_games = load_chess_dataset(CSV_PATH, MIN_ELO)

    # Part 2: Extract training pairs
    df_training = extract_training_pairs(df_games, max_games=MAX_GAMES)

    # Part 3: Create encodings and dataset
    print("\n" + "=" * 60)
    print("PART 3: Creating Encodings and Dataset")
    print("=" * 60)

    uci_to_idx, idx_to_uci = create_move_mappings()
    print(f"\nTotal possible moves (classes): {len(uci_to_idx):,}")

    dataset = ChessDataset(df_training, uci_to_idx)

    # Split into train/validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nTrain set size: {len(train_dataset):,}")
    print(f"Validation set size: {len(val_dataset):,}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Part 4: Build and train model
    num_classes = len(uci_to_idx)
    model = ChessCNN(num_classes=num_classes)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Part 5: Save model and mappings
    save_model(model, uci_to_idx, idx_to_uci)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nFiles saved:")
    print("  - model.pth (model weights)")
    print("  - move_mappings.json (move encoding)")

if __name__ == "__main__":
    main()
