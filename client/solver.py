# -*- coding: utf-8 -*-
import sys
import pygame
import time
import math
import config
import ws_client

g_useRender = True

BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
CORAL = (255, 157, 0)
GREY = (204, 204, 204)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class BlockType:
    EMPTY = 0
    NON_BREAKABLE = 1
    BREAKABLE = 2
    LEDDER = 3
    GOLD = 4
    ME = 5
    AI = 6

legend = {
    u' ': (BlockType.EMPTY, None),
    u"☼": (BlockType.NON_BREAKABLE, CORAL),
    u'#': (BlockType.BREAKABLE, ORANGE),
    u'H': (BlockType.LEDDER, GREY),
    u'$': (BlockType.GOLD, YELLOW),
    
    u'Ѡ': (BlockType.ME, RED),
    u'Я': (BlockType.ME, RED),
    u'R': (BlockType.ME, RED),
    u'Y': (BlockType.ME, RED),
    u'◄': (BlockType.ME, RED),
    u'►': (BlockType.ME, RED),
    u']': (BlockType.ME, RED),
    u'[': (BlockType.ME, RED),
    u'}': (BlockType.ME, RED),
    u'{': (BlockType.ME, RED),

    u'Q': (BlockType.AI, GREEN),
    u'«': (BlockType.AI, GREEN),
    u'»': (BlockType.AI, GREEN),
    u'<': (BlockType.AI, GREEN),
    u'<': (BlockType.AI, GREEN),
    u'X': (BlockType.AI, GREEN),
}


class Map():

    def __init__(self):
        self.matrix = None
        self.size = 0

    def parseMsg(self, msg):
        msg_len = len(msg)
        sqrt_len = int(math.sqrt(msg_len))
        self.size = sqrt_len
        self.matrix = [
            [None for x in range(sqrt_len)] for x in range(sqrt_len)]

        for c in range(0, msg_len):
            if msg[c] not in legend.keys():
                continue

            x = c % sqrt_len
            y = c // sqrt_len
            self.matrix[x][y] = legend[msg[c]]


class Unit(object):

    def __init__(self):
        self.x = 50
        self.y = 50
        self.size = 20


class Solver():

    def __init__(self):
        print "Inited"

    def solve(self, map):
        res = 'RIGHT'
        return res


class Render():

    def __init__(self):
        if g_useRender: 
            pygame.init()
            self.m_window = pygame.display.set_mode((640, 640))
            print "Render inited"
        else:
            print "Render is NOT used"

    def DrawMap(self, map):
        if not str or not g_useRender:
            return

        if not map.matrix:
            return

        self.m_window.fill((0, 0, 0))

        for i in range(map.size):
            for j in range(map.size):
                if map.matrix[i][j] and map.matrix[i][j][1]:
                    pygame.draw.circle(
                        self.m_window, map.matrix[i][j][1], [10 * i, 10 * j], 5)
        # pygame.draw.circle(self.m_window, BLUE, [100, 100], 20)
        pygame.display.flip()

    def start(self):
        msg = None
        map = Map()
        solver = Solver()
        while True:
            time.sleep(0.1)
            if g_useRender:
                for event in pygame.event.get():
                    a = 1
            if config.newMSG:
                msg = ws_client.AccessVarMSG(True, None)
                map.parseMsg(msg)
                turn = solver.solve(map)
                ws_client.AccessVarResult(False, turn)

            self.DrawMap(map)
