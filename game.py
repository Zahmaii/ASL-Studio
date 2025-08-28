# game.py
import pygame
import random
import sys

pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ASL Sign Matching Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)

# Fonts
font = pygame.font.SysFont("Arial", 60)
small_font = pygame.font.SysFont("Arial", 30)

# Letters A–Z
letters = [chr(i) for i in range(65, 91)]  # A–Z
target_letter = random.choice(letters)

score = 0
timer = 30  # seconds
clock = pygame.time.Clock()

# Game loop
start_ticks = pygame.time.get_ticks()

running = True
while running:
    screen.fill(WHITE)

    # Time countdown
    seconds = timer - (pygame.time.get_ticks() - start_ticks) // 1000
    if seconds <= 0:
        running = False

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.unicode.upper() == target_letter:
                score += 1
                target_letter = random.choice(letters)

    # Draw UI
    target_text = font.render(f"Show Sign: {target_letter}", True, BLACK)
    screen.blit(target_text, (WIDTH // 2 - target_text.get_width() // 2, HEIGHT // 3))

    score_text = small_font.render(f"Score: {score}", True, GREEN)
    screen.blit(score_text, (20, 20))

    time_text = small_font.render(f"Time: {seconds}", True, RED)
    screen.blit(time_text, (WIDTH - 150, 20))

    pygame.display.flip()
    clock.tick(30)

# Game over screen
screen.fill(WHITE)
end_text = font.render("Game Over!", True, RED)
score_text = small_font.render(f"Your Score: {score}", True, BLACK)
screen.blit(end_text, (WIDTH // 2 - end_text.get_width() // 2, HEIGHT // 3))
screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
pygame.display.flip()

pygame.time.wait(5000)
pygame.quit()
