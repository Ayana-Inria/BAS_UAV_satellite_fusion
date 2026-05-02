"""Default constants used by the fusion package."""

N_CLASSES = 2
WINDOW_SIZE = (480, 480)
BATCH_SIZE = 1

LABELS = ["burnt", "unburnt"]
PALETTE = {
    0: (255, 0, 0),
    1: (0, 255, 0),
}
INVERT_PALETTE = {v: k for k, v in PALETTE.items()}
