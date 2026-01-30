import pygame
import sys

pygame.init()

# You must create a window for pygame to receive keyboard events
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Keyboard test")

clock = pygame.time.Clock()

running = True
while running:
    # 1. Process events every frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # 2. Check for keydown events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("ESC pressed, quitting.")
                running = False
            elif event.key == pygame.K_w:
                print("W pressed")
            elif event.key == pygame.K_a:
                print("A pressed")
            elif event.key == pygame.K_s:
                print("S pressed")
            elif event.key == pygame.K_d:
                print("D pressed")

    # You can also poll the current key state:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        # This prints while space is *held down*
        print("Holding SPACE")

    # Just fill window so it doesn't look frozen
    screen.fill((0, 0, 0))
    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS

pygame.quit()
sys.exit()

