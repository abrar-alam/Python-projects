import pygame
import os
import time
import random
import random
pygame.font.init()
from pygame.time import Clock

WIDTH, HEIGHT = 750, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Invader")

# Load all the image assets

sourceFileDir = os.path.dirname(os.path.abspath(__file__))

RED_SPACE_SHIP = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_ship_red_small.png"))
BLUE_SPACE_SHIP = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_ship_blue_small.png"))
GREEN_SPACE_SHIP = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_ship_green_small.png"))
BLUE_SPACE_SHIP = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_ship_blue_small.png"))

# This is user's space ship
YELLOW_SPACE_SHIP = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_ship_yellow.png"))

# Load  waepons
RED_LASER = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_laser_red.png"))
GREEN_LASER = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_laser_green.png"))
BLUE_LASER = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_laser_blue.png"))
YELLOW_LASER = pygame.image.load(
    os.path.join(sourceFileDir, "assets", "pixel_laser_yellow.png"))

# Load background
BG = pygame.transform.scale(pygame.image.load(
    os.path.join(sourceFileDir, "assets", "background-black.png")), (WIDTH, HEIGHT))

class Ship:
    def __init__(self, x, y, health =100):
        self.x = x
        self.y = y
        self.health = health
        self.ship_img = None
        self.laser_img = None
        self.lasers = []
        self.cool_down_counter = 0

    def draw(self, window):
        pygame.draw.rect(window, (255,0,0), (self.x, self.y, 50, 50))



def main():
    run = True
    FPS = 60
    level = 1
    lives = 5
    main_font = pygame.font.SysFont("comicsans", 50)

    ship = Ship(300, 650)

    clock = pygame.time.Clock()

    def redraw_window():
        WIN.blit(BG, (0,0))
        # Draw text
        lives_label = main_font.render(f"Lives: {lives}", 1, (255, 255, 255))
        level_label = main_font.render(f"Level: {level}", 1, (255, 255, 255))

        WIN.blit(lives_label, (10,10))
        WIN.blit(level_label, (WIDTH - level_label.get_width()- 10, 10))

        ship.draw(WIN)
        # Refreshes the display
        pygame.display.update()

    while run:
        clock.tick(FPS)
        redraw_window()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: # left
            ship.x-=1
            
main()
