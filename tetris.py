import pygame
import random

BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

TETROMINOS = [
    # I (Cyan)
    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],
    
    # O (Yellow)
    [[1, 1],
     [1, 1]],
    
    # T (Magenta)
    [[0, 1, 0],
     [1, 1, 1],
     [0, 0, 0]],
    
    # S (Green)
    [[0, 1, 1],
     [1, 1, 0],
     [0, 0, 0]],
    
    # Z (Red)
    [[1, 1, 0],
     [0, 1, 1],
     [0, 0, 0]],
    
    # J (Blue)
    [[1, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],

    # L (Orange)
    [[0, 0, 1],
     [1, 1, 1],
     [0, 0, 0]]
]
TETROMINO_COLORS = [CYAN, YELLOW, MAGENTA, GREEN, RED, BLUE, ORANGE]

class Tetris:
    def __init__(self, grid_width=10, grid_height=20, cell_size=30, use_bag_randomizer=True):
        pygame.init()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.screen_size_height = grid_height * self.cell_size
        self.screen_size_width = grid_width * self.cell_size
        
        self.surface = pygame.Surface((self.screen_size_width, self.screen_size_height))

        self.tetris_grid = [[BLACK for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.use_bag_randomizer = use_bag_randomizer
        self.bag = []
        self.current_tetromino = self.get_tetromino()
        self.is_game_over = False
        self.score = 0
        self.tick_count = 0
        
        self.draw_state()
    
    def get_tetromino(self):
        if self.use_bag_randomizer:
            if not self.bag:
                self.bag = list(range(len(TETROMINOS)))
                random.shuffle(self.bag)
            tetromino_index = self.bag.pop()
        else:
            tetromino_index = random.randint(0, len(TETROMINOS) - 1)
        tetromino_shape = TETROMINOS[tetromino_index]
        return {
            'shape': tetromino_shape,
            'color': TETROMINO_COLORS[tetromino_index],
            'x': (self.grid_width - len(tetromino_shape[0])) // 2,
            'y': 0 
        }
    
    def draw_state(self):
        self.surface.fill(BLACK)
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                pygame.draw.rect(self.surface, self.tetris_grid[y][x], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.surface, GRAY, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1)

        for y, row in enumerate(self.current_tetromino['shape']):
            for x, cell in enumerate(row):
                if cell:
                    offset_x = (self.current_tetromino['x'] + x)
                    offset_y = (self.current_tetromino['y'] + y)
                    pygame.draw.rect(self.surface, self.current_tetromino['color'], (offset_x * self.cell_size, offset_y * self.cell_size, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.surface, GRAY, (offset_x * self.cell_size, offset_y * self.cell_size, self.cell_size, self.cell_size), 1)

    def get_state_image(self):
        return self.surface

    def process_action(self, action):
        if action == 'LEFT':
            if self.is_action_valid(self.current_tetromino, -1, 0):
                self.current_tetromino['x'] -= 1
        elif action == 'RIGHT':
            if self.is_action_valid(self.current_tetromino, 1, 0):
                self.current_tetromino['x'] += 1
        elif action == 'ROTATE':
            rotated_tetromino_shape = self.rotate_tetromino_shape(self.current_tetromino['shape'])
            if self.is_action_valid(self.current_tetromino, 0, 0, rotated_tetromino_shape):
                self.current_tetromino['shape'] = rotated_tetromino_shape

        if self.is_action_valid(self.current_tetromino, 0, 1):
            self.current_tetromino['y'] += 1
        else:
            self.lock_tetromino()

        self.draw_state()

        self.tick_count += 1
        
    def is_action_valid(self, tetromino, offset_x, offset_y, rotated_tetromino_shape=None):
        tetromino_shape = rotated_tetromino_shape if rotated_tetromino_shape else tetromino['shape']
        for y, row in enumerate(tetromino_shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = tetromino['x'] + x + offset_x
                    new_y = tetromino['y'] + y + offset_y
                    if new_x < 0 or new_x >= self.grid_width or new_y >= self.grid_height or self.tetris_grid[new_y][new_x] != BLACK:
                        return False
        return True

    def rotate_tetromino_shape(self, tetromino_shape):
        return [list(row) for row in zip(*tetromino_shape[::-1])]
    
    def lock_tetromino(self):
        for y, row in enumerate(self.current_tetromino['shape']):
            for x, cell in enumerate(row):
                if cell:
                    self.tetris_grid[self.current_tetromino['y'] + y][self.current_tetromino['x'] + x] = self.current_tetromino['color']
        self.try_line_clear()
        self.current_tetromino = self.get_tetromino()
        if not self.is_action_valid(self.current_tetromino, 0, 0):
            self.is_game_over = True

    def try_line_clear(self):
        new_tetris_grid = [row for row in self.tetris_grid if any(cell == BLACK for cell in row)]
        lines_cleared = self.grid_height - len(new_tetris_grid)
        score_multipliers = [100, 300, 500, 800]
        if lines_cleared > 0:
            self.score += score_multipliers[lines_cleared - 1]
        for _ in range(lines_cleared):
            new_tetris_grid.insert(0, [BLACK for _ in range(self.grid_width)])
        self.tetris_grid = new_tetris_grid
