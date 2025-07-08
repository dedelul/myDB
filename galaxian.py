import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Galaxian')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Game settings
enemy_rows = 5
enemy_cols = 10
enemy_size = 40
player_size = 50
bullet_size = 5

clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((player_size, player_size))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH // 2
        self.rect.bottom = HEIGHT - 10
        self.speed = 5

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        # keep within screen
        self.rect.x = max(0, min(WIDTH - self.rect.width, self.rect.x))

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((enemy_size, enemy_size))
        self.image.fill(RED)
        self.rect = self.image.get_rect(topleft=(x, y))

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((bullet_size, bullet_size * 2))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = -7

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0:
            self.kill()

# Sprite groups
player_group = pygame.sprite.GroupSingle()
player = Player()
player_group.add(player)

enemy_group = pygame.sprite.Group()
bullet_group = pygame.sprite.Group()

# Create enemies
for row in range(enemy_rows):
    for col in range(enemy_cols):
        x = 100 + col * (enemy_size + 10)
        y = 50 + row * (enemy_size + 10)
        enemy = Enemy(x, y)
        enemy_group.add(enemy)

enemy_direction = 1
enemy_speed = 1

# Main loop
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bullet = Bullet(player.rect.centerx, player.rect.top)
                bullet_group.add(bullet)

    keys = pygame.key.get_pressed()
    player_group.update(keys)
    bullet_group.update()

    # Move enemies
    move_down = False
    for enemy in enemy_group:
        enemy.rect.x += enemy_direction * enemy_speed
        if enemy.rect.right >= WIDTH or enemy.rect.left <= 0:
            move_down = True
    if move_down:
        enemy_direction *= -1
        for enemy in enemy_group:
            enemy.rect.y += enemy_size // 2

    # Collision detection
    for bullet in pygame.sprite.groupcollide(bullet_group, enemy_group, True, True):
        pass

    # Draw
    screen.fill(BLACK)
    player_group.draw(screen)
    enemy_group.draw(screen)
    bullet_group.draw(screen)
    pygame.display.flip()

pygame.quit()
sys.exit()
