"""
llm_tetris_eval.py
------------------
Runs multiple LLMs against the Tetris environment and records evaluation metrics.
Uses a pure-text board renderer — no pygame display required.

Requirements:
    pip install openai pandas matplotlib pygame-ce
    (pygame-ce is a community fork; tetris.py needs it internally but we never
     open a window ourselves.)

Usage:
    export OPENROUTER_API_KEY="key"
    python llm_tetris_eval.py

Outputs (written to ./results/):
    tetris_results.csv   — one row per game
    tetris_summary.csv   — per-model aggregate stats
    tetris_eval.png      — comparison plots
"""

import os
import time
import random
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import openai
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

#  Tetris import 
# pygame is only used internally by tetris.py — we never open a window.
import pygame
pygame.init()

from tetris import Tetris, TETROMINOS, TETROMINO_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR         = "results"
N_GAMES             = 1        # games per model
MAX_TICKS           = 10      # hard cap per game
TICK_DELAY          = 5.0      # seconds between ticks (>0 to slow down)

# Replace these two lines at the top of the config
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV         = "OPENROUTER_API_KEY"

"""
curl https://openrouter.ai/api/v1/models | python3 -c "
import json, sys
models = json.load(sys.stdin)['data']
free = [m['id'] for m in models if m.get('pricing', {}).get('prompt') == '0']
print('\n'.join(sorted(free)))
"
"""

MODELS = [
    # ("Nemotron-3-Nano-30B",  "nvidia/nemotron-3-nano-30b-a3b:free"),
    # ("Nemotron-3-Super-120B", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("Nemotron-Nano-12B-VL", "nvidia/nemotron-nano-12b-v2-vl:free"),
    # ("Nemotron-Nano-9B",     "nvidia/nemotron-nano-9b-v2:free"),
]

VALID_ACTIONS = {"LEFT", "RIGHT", "ROTATE", "HOLD", "DOWN"}
PIECE_NAMES   = ["I", "O", "T", "S", "Z", "J", "L"]

# ─────────────────────────────────────────────────────────────────────────────
# TEXT BOARD RENDERER
# ─────────────────────────────────────────────────────────────────────────────

_BLACK    = (0, 0, 0)
_CHAR_MAP = {
    (0,   255, 255): "I",
    (255, 255, 0):   "O",
    (255, 0,   255): "T",
    (0,   255, 0):   "S",
    (255, 0,   0):   "Z",
    (0,   0,   255): "J",
    (255, 165, 0):   "L",
}


def _ghost_y(tetris: Tetris) -> int:
    offset = 0
    while tetris.is_action_valid(tetris.current_tetromino, 0, offset + 1):
        offset += 1
    return tetris.current_tetromino["y"] + offset


def render_board_text(tetris: Tetris) -> str:
    """
    Render the board as ASCII art, e.g.:

        HOLD: T
        ┌──────────┐
        │..........│
        │....i.....│   ← ghost (lowercase)
        │..TTTT....│   ← current piece (uppercase)
        │JJJSSZZ...│   ← locked cells
        └──────────┘
        CURRENT: S
        SCORE: 300   TICKS: 42
        VALID ACTIONS: DOWN HOLD LEFT RIGHT ROTATE
    """
    w, h = tetris.grid_width, tetris.grid_height
    cur  = tetris.current_tetromino

    # Locked cells
    grid = [
        [_CHAR_MAP.get(tuple(cell), ".") if tuple(cell) != _BLACK else "."
         for cell in row]
        for row in tetris.tetris_grid
    ]

    # Ghost (lowercase)
    ghost_y   = _ghost_y(tetris)
    ghost_chr = PIECE_NAMES[cur["index"]].lower()
    for dy, row in enumerate(cur["shape"]):
        for dx, cell in enumerate(row):
            if cell:
                gy, gx = ghost_y + dy, cur["x"] + dx
                if 0 <= gy < h and 0 <= gx < w and grid[gy][gx] == ".":
                    grid[gy][gx] = ghost_chr

    # Current piece (uppercase)
    cur_chr = PIECE_NAMES[cur["index"]]
    for dy, row in enumerate(cur["shape"]):
        for dx, cell in enumerate(row):
            if cell:
                cy, cx = cur["y"] + dy, cur["x"] + dx
                if 0 <= cy < h and 0 <= cx < w:
                    grid[cy][cx] = cur_chr

    hold_label = (
        PIECE_NAMES[tetris.held_tetromino_index]
        if tetris.held_tetromino_index is not None else "none"
    )

    lines = [
        f"HOLD: {hold_label}",
        "┌" + "─" * w + "┐",
        *("│" + "".join(row) + "│" for row in grid),
        "└" + "─" * w + "┘",
        f"CURRENT: {cur_chr}",
        f"SCORE: {tetris.score}   TICKS: {tetris.tick_count}",
        f"VALID ACTIONS: {' '.join(sorted(VALID_ACTIONS))}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Tetris player. You receive the current board as ASCII art.

Legend:
  .               = empty cell
  I O T S Z J L  = locked piece of that type (uppercase letter)
  i o t s z j l  = ghost — preview of where the current piece will land
  CURRENT         = the falling piece you control
  HOLD            = piece saved for later

Respond with EXACTLY ONE of these words and nothing else:
  LEFT    move current piece one cell left
  RIGHT   move current piece one cell right
  ROTATE  rotate current piece 90° clockwise
  HOLD    swap current piece with held piece (only once per piece)
  DOWN    do nothing / let the piece fall one row

Strategy tips:
- Keep the stack as flat and as low as possible.
- Fill rows fully to clear lines and score points.
- Avoid creating holes (empty cells with filled cells above them).
- Save the I piece in HOLD for Tetris (4-line clear).
- Use the ghost piece (lowercase) to plan where the piece will land.

Output ONLY the single action word. No explanation, punctuation, or other text.
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GameRecord:
    model_name:       str
    game_index:       int
    score:            int   = 0
    ticks_survived:   int   = 0
    lines_cleared:    int   = 0
    actions_taken:    dict  = field(default_factory=lambda: defaultdict(int))
    invalid_actions:  int   = 0
    api_errors:       int   = 0
    total_latency_ms: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_model(client: openai.OpenAI, model_string: str, board_text: str) -> tuple[str, float]:
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_string,
        max_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": board_text},
        ],
    )
    latency_ms = (time.time() - t0) * 1000
    
    # Guard against None content or empty choices
    if not response.choices:
        return "", latency_ms
    content = response.choices[0].message.content
    return content if content is not None else "", latency_ms

def parse_action(raw: str) -> Optional[str]:
    if not raw:
        return None
    # Strip reasoning/thinking blocks
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Scan all words for a valid action, not just the first
    for word in raw.upper().split():
        if word in VALID_ACTIONS:
            return word
    return None

# ─────────────────────────────────────────────────────────────────────────────
# GAME LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_game(
    client:       openai.OpenAI,
    model_name:   str,
    model_string: str,
    game_index:   int,
) -> GameRecord:
    record     = GameRecord(model_name=model_name, game_index=game_index)
    tetris     = Tetris(can_hold_tetromino=True, use_bag_randomization=True)
    prev_score = 0

    print(f"  Game {game_index + 1}/{N_GAMES} | {model_name}")

    while not tetris.is_game_over and record.ticks_survived < MAX_TICKS:
        board_text = render_board_text(tetris)

        try:
            raw, latency = _call_model(client, model_string, board_text)
            record.total_latency_ms += latency
        except Exception as exc:
            print(f"    [API ERROR] {exc}")
            record.api_errors += 1
            raw = "DOWN"

        action = parse_action(raw)
        if action is None:
            print(f"    [INVALID] {repr(raw)} → DOWN")
            record.invalid_actions += 1
            action = "DOWN"

        record.actions_taken[action] += 1
        tetris.process_action(action)
        record.ticks_survived += 1

        if tetris.score > prev_score:
            delta = tetris.score - prev_score
            record.lines_cleared += {100: 1, 300: 2, 500: 3, 800: 4}.get(delta, 0)
            prev_score = tetris.score

        if TICK_DELAY > 0:
            time.sleep(TICK_DELAY)

    record.score = tetris.score
    print(
        f"    → score={record.score}  ticks={record.ticks_survived}  "
        f"lines={record.lines_cleared}  invalid={record.invalid_actions}  "
        f"errors={record.api_errors}"
    )
    return record

# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(models: list[tuple[str, str]], n_games: int = N_GAMES) -> list[GameRecord]:
    client = openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ[API_KEY_ENV],
    )
    all_records: list[GameRecord] = []

    for display_name, model_string in models:
        print(f"\n{'='*60}")
        print(f"  {display_name}  ({model_string})")
        print(f"{'='*60}")
        for g in range(n_games):
            try:
                rec = run_game(client, display_name, model_string, g)
            except Exception as exc:
                print(f"  [GAME CRASHED] {exc}")
                traceback.print_exc()
                rec = GameRecord(model_name=display_name, game_index=g, api_errors=1)
            all_records.append(rec)

    return all_records

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS → CSV
# ─────────────────────────────────────────────────────────────────────────────

def records_to_dataframe(records: list[GameRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "model":           r.model_name,
            "game":            r.game_index,
            "score":           r.score,
            "ticks_survived":  r.ticks_survived,
            "lines_cleared":   r.lines_cleared,
            "invalid_actions": r.invalid_actions,
            "api_errors":      r.api_errors,
            "avg_latency_ms":  r.total_latency_ms / max(r.ticks_survived, 1),
        }
        for a in VALID_ACTIONS:
            row[f"action_{a.lower()}"] = r.actions_taken.get(a, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(
            mean_score       =("score",           "mean"),
            std_score        =("score",           "std"),
            max_score        =("score",           "max"),
            mean_ticks       =("ticks_survived",  "mean"),
            mean_lines       =("lines_cleared",   "mean"),
            total_invalid    =("invalid_actions", "sum"),
            total_api_errors =("api_errors",      "sum"),
            mean_latency_ms  =("avg_latency_ms",  "mean"),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if API_KEY_ENV not in os.environ:
        raise EnvironmentError(
            f"Set the {API_KEY_ENV} environment variable before running.\n"
            f"  export {API_KEY_ENV}=your_key_here"
        )

    records = evaluate_models(MODELS, n_games=N_GAMES)
    df      = records_to_dataframe(records)
    summary = build_summary(df)

    df.to_csv(os.path.join(RESULTS_DIR, "tetris_results.csv"),  index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "tetris_summary.csv"), index=False)

    print("\n\n── SUMMARY ──────────────────────────────────────────")
    print(summary.to_string(index=False))


    print(f"\nRaw results  → {RESULTS_DIR}/tetris_results.csv")
    print(f"Summary      → {RESULTS_DIR}/tetris_summary.csv")


if __name__ == "__main__":
    main()