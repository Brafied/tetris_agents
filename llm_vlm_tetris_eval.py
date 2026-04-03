import os
import time
import random
import traceback
import base64
import mimetypes
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import openai
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygame
pygame.init()

from tetris import Tetris, TETROMINOS, TETROMINO_COLORS
from manual_tests import SNAPSHOTS_DIRECTORY

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR         = "results"
N_GAMES             = 1
MAX_TICKS           = 10
TICK_DELAY          = 5.0

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV         = "OPENROUTER_API_KEY"

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    uses_image: bool = False

MODELS = [
    ModelSpec("Nemotron-Nano-12B-VL", "nvidia/nemotron-nano-12b-v2-vl:free", uses_image=True),
    # Add more vision models:
    # ModelSpec("Qwen2.5-VL-7B", "qwen/qwen2.5-vl-7b-instruct:free", uses_image=True),
    # ModelSpec("Llama-3.2-11B-Vision", "meta-llama/llama-3.2-11b-vision-instruct:free", uses_image=True),

    # Text-only models:
    # ModelSpec("Qwen2.5-7B-Instruct", "qwen/qwen2.5-7b-instruct:free", uses_image=False),
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
    w, h = tetris.grid_width, tetris.grid_height
    cur  = tetris.current_tetromino

    grid = [
        [_CHAR_MAP.get(tuple(cell), ".") if tuple(cell) != _BLACK else "."
         for cell in row]
        for row in tetris.tetris_grid
    ]

    ghost_y   = _ghost_y(tetris)
    ghost_chr = PIECE_NAMES[cur["index"]].lower()
    for dy, row in enumerate(cur["shape"]):
        for dx, cell in enumerate(row):
            if cell:
                gy, gx = ghost_y + dy, cur["x"] + dx
                if 0 <= gy < h and 0 <= gx < w and grid[gy][gx] == ".":
                    grid[gy][gx] = ghost_chr

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
# SNAPSHOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

def find_snapshot(tick_index: int) -> Optional[Path]:
    """
    saved as tick_{tetris.tick_count:04d}.png
    """
    base = Path(SNAPSHOTS_DIRECTORY)

    candidates = [
        base / f"tick_{tick_index:04d}.png",
    ]
    for p in candidates:
        if p.exists():
            return p

    if base.exists() and base.is_dir():
        imgs = sorted([p for p in base.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
        if 0 <= tick_index < len(imgs):
            return imgs[tick_index]

    return None

def image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def build_vlm_user_content(board_text: str, snapshot_path: Path) -> list[dict]:
    """
    Sends both the text state and the snapshot.
    The model should use the snapshot as the primary source.
    """
    return [
        {
            "type": "text",
            "text": (
                "Choose the next Tetris action from this state.\n\n"
                f"{board_text}\n\n"
                "Use the snapshot to inspect the board."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(snapshot_path)
            },
        },
    ]

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEXT = """\
You are an expert Tetris player. You receive the current board as ASCII art.

Legend:
  .               = empty cell
  I O T S Z J L  = locked piece of that type (uppercase letter)
  i o t s z j l  = ghost — preview of where the current piece will land
  CURRENT         = the falling piece you control
  HOLD            = piece saved for later

Respond with EXACTLY ONE of these words and nothing else:
  LEFT
  RIGHT
  ROTATE
  HOLD
  DOWN

Output ONLY the single action word. No explanation, punctuation, or other text.
"""

SYSTEM_PROMPT_VLM = """\
You are an expert Tetris player.

You will receive:
1. Text metadata about the board
2. A snapshot of the board

Respond with EXACTLY ONE of these words and nothing else:
  LEFT
  RIGHT
  ROTATE
  HOLD
  DOWN

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
# MODEL CALLS
# ─────────────────────────────────────────────────────────────────────────────

def _call_text_model(client: openai.OpenAI, model_string: str, board_text: str) -> tuple[str, float]:
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_string,
        max_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user",   "content": board_text},
        ],
    )
    latency_ms = (time.time() - t0) * 1000
    if not response.choices:
        return "", latency_ms
    content = response.choices[0].message.content
    return content if content is not None else "", latency_ms

def _call_vlm(
    client: openai.OpenAI,
    model_string: str,
    board_text: str,
    snapshot_path: Path,
) -> tuple[str, float]:
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_string,
        max_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_VLM},
            {"role": "user", "content": build_vlm_user_content(board_text, snapshot_path)},
        ],
    )
    latency_ms = (time.time() - t0) * 1000
    if not response.choices:
        return "", latency_ms
    content = response.choices[0].message.content
    return content if content is not None else "", latency_ms

def parse_action(raw: str) -> Optional[str]:
    if not raw:
        return None
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
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
    uses_image:   bool,
    game_index:   int,
) -> GameRecord:
    record    = GameRecord(model_name=model_name, game_index=game_index)
    tetris    = Tetris(can_hold_tetromino=True, use_bag_randomization=True)
    prev_score = 0

    print(f"  Game {game_index + 1}/{N_GAMES} | {model_name}")

    while not tetris.is_game_over and record.ticks_survived < MAX_TICKS:
        board_text = render_board_text(tetris)

        try:
            if uses_image:
                snapshot_path = find_snapshot(record.ticks_survived)
                if snapshot_path is None:
                    print("    [NO SNAPSHOT FOUND] falling back to text board")
                    raw, latency = _call_text_model(client, model_string, board_text)
                else:
                    raw, latency = _call_vlm(client, model_string, board_text, snapshot_path)
            else:
                raw, latency = _call_text_model(client, model_string, board_text)

            record.total_latency_ms += latency
        except Exception as exc:
            print(f"    [API ERROR] {exc}")
            record.api_errors += 1
            raw = "DOWN"

        action = parse_action(raw)
        if action is None:
            print(f"    [INVALID] {repr(raw)} -> DOWN")
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
        f"    -> score={record.score}  ticks={record.ticks_survived}  "
        f"lines={record.lines_cleared}  invalid={record.invalid_actions}  "
        f"errors={record.api_errors}"
    )
    return record

# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(models: list[ModelSpec], n_games: int = N_GAMES) -> list[GameRecord]:
    client = openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ[API_KEY_ENV],
    )
    all_records: list[GameRecord] = []

    for spec in models:
        print(f"\n{'='*60}")
        print(f"  {spec.name}  ({spec.model_id})")
        print(f"{'='*60}")
        for g in range(n_games):
            try:
                rec = run_game(client, spec.name, spec.model_id, spec.uses_image, g)
            except Exception as exc:
                print(f"  [GAME CRASHED] {exc}")
                traceback.print_exc()
                rec = GameRecord(model_name=spec.name, game_index=g, api_errors=1)
            all_records.append(rec)

    return all_records

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS -> CSV
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
            mean_ticks       =("ticks_survived",   "mean"),
            mean_lines       =("lines_cleared",    "mean"),
            total_invalid    =("invalid_actions",  "sum"),
            total_api_errors =("api_errors",      "sum"),
            mean_latency_ms  =("avg_latency_ms",   "mean"),
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

    df.to_csv(os.path.join(RESULTS_DIR, "tetris_results.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "tetris_summary.csv"), index=False)

    print("\n\n-- SUMMARY ------------------------------------------------")
    print(summary.to_string(index=False))

    print(f"\nRaw results  -> {RESULTS_DIR}/tetris_results.csv")
    print(f"Summary      -> {RESULTS_DIR}/tetris_summary.csv")

if __name__ == "__main__":
    main()