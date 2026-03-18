from tetris import Tetris
import os, pygame

SNAPSHOTS_DIRECTORY = "snapshots"

def play_with_image():
    if not os.path.exists(SNAPSHOTS_DIRECTORY):
        os.makedirs(SNAPSHOTS_DIRECTORY)

    tetris = Tetris()

    while not tetris.is_game_over:
        snapshot = tetris.get_state_image()
        filename = os.path.join(SNAPSHOTS_DIRECTORY, f"tick_{tetris.tick_count:04d}.png")
        pygame.image.save(snapshot, filename)
        
        action = input(f"Action > ")
        tetris.process_action(action)

    print(f"Final Score: {tetris.score}")

def play_with_window():
    tetris = Tetris()

    screen = pygame.display.set_mode(
        (tetris.surface.get_width(), tetris.surface.get_height())
    )

    while not tetris.is_game_over:
        screen.blit(tetris.get_state_image(), (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    tetris.process_action("HOLD")
                elif event.key == pygame.K_LEFT:
                    tetris.process_action("LEFT")
                elif event.key == pygame.K_RIGHT:
                    tetris.process_action("RIGHT")
                elif event.key == pygame.K_UP:
                    tetris.process_action("ROTATE")
                elif event.key == pygame.K_DOWN:
                    tetris.process_action(None)

    print(f"Final Score: {tetris.score}")

if __name__ == "__main__":
    #play_with_image()
    play_with_window()

