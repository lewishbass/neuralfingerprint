

import numpy as np
import time
import pygame
from molvis import Molvis
import pickle

from neuralfingerprint import MolFrom

d = pickle.load(open('consol.pickle2', 'rb'))
pygame.init()
font = pygame.font.Font('freesansbold.ttf', 20)
size = (500, 500)
screen = pygame.display.set_mode(size, pygame.RESIZABLE)
mvis = Molvis((500, 500), (0, 0), MolFrom('mobley_2661134'))

indx=0
running = True
t = time.time()
ml = np.array([0, 0])

while(running):
    ft = time.time()-t
    t = time.time()
    for event in pygame.event.get():
        if(event.type == pygame.QUIT or event.type == pygame.KEYDOWN):
            running = False
        if(event.type == pygame.MOUSEWHEEL):
            indx = indx+event.y
            i = d.keys()[indx%len(d.keys())]
            mvis.reMol(i, d[i])
        if (event.type == pygame.VIDEORESIZE):
            mvis.resize(screen.get_size(), (0, 0))

    mn = np.array(pygame.mouse.get_pos())
    if(pygame.mouse.get_pressed()[0]):
        mvis.rotMol((mn-ml)[0], (mn-ml)[1])
    ml = mn

    screen.fill((0, 0, 0))
    mvis.draw(screen, font, ft)
    pygame.display.flip()

pygame.quit()