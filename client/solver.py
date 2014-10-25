# -*- coding: utf-8 -*-
import sys
import pygame
import time
import math
import config
import ws_client

g_useRender = False

BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
CORAL = (255, 157, 0)
GREY = (204, 204, 204)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 50)
WHITE = (153, 255, 255)

class BlockType:
    EMPTY = 0
    NON_BREAKABLE = 1
    BREAKABLE = 2
    LADDER = 3
    GOLD = 4
    ME = 5
    AI = 6
    PIPE = 7
    ENEMY = 8 

f = open('dump_Graph.txt','w')

legend = {
    u' ': (BlockType.EMPTY, None),
    
    u'.': (BlockType.NON_BREAKABLE, CORAL),
    u'*': (BlockType.NON_BREAKABLE, CORAL),
    u"☼": (BlockType.NON_BREAKABLE, CORAL),

    u'#': (BlockType.BREAKABLE, ORANGE),
    u'1': (BlockType.BREAKABLE, ORANGE),
    u'2': (BlockType.BREAKABLE, ORANGE),
    u'3': (BlockType.BREAKABLE, ORANGE),
    u'4': (BlockType.BREAKABLE, ORANGE),
    

    u'H': (BlockType.LADDER, GREY),
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
    u'>': (BlockType.AI, GREEN),
    u'X': (BlockType.AI, GREEN),

    u'~': (BlockType.PIPE, BLUE),

    u'Z': (BlockType.ENEMY, WHITE),
    u')': (BlockType.ENEMY, WHITE),
    u'(': (BlockType.ENEMY, WHITE),
    u'U': (BlockType.ENEMY, WHITE),
    u'Э': (BlockType.ENEMY, WHITE),
    u'Є': (BlockType.ENEMY, WHITE),

}


class Joint():
    def __init__(self, point):
        self.point = point
        self.val = None

class Node():
    def __init__(self, block_type):
        self.type = block_type
        self.joints = []

    def addJoint(self, joint):
        self.joints.append(joint)

        
class Map():
    def __init__(self):
        self.matrix = None
        self.size = 0
        self.me_point = None

    def parseMsg(self, msg):
        msg_len = len(msg)
        sqrt_len = int(math.sqrt(msg_len))
        self.size = sqrt_len
        self.matrix = [
            [None for x in range(sqrt_len)] for x in range(sqrt_len)]

        for c in range(0, msg_len):
            if msg[c] not in legend.keys():
                print "WARNING: unknow symbol: "
                print msg[c].encode('utf-8')
                continue

            x = c % sqrt_len
            y = c // sqrt_len

            tmp = Node(legend[msg[c]])
            if tmp.type[0] == BlockType.ME:
                self.me_point = (x,y)

            if msg[c] == u'Y':
                node = Node(legend[u'H'])
            if msg[c] == u'{' or msg[c] == u'}':
                node = Node(legend[u'~'])
            else:
                node = Node(legend[msg[c]])
            self.matrix[x][y] = node


    def createGraphInfo(self):
        for i in range(self.size):
            for j in range(self.size):
                curr = self.matrix[i][j]
                if curr is None:
                    print "WARNING map no element"
                    continue
                
                #create joints
                
                curr_left = self.GetLeftNode(i,j)
                curr_right = self.GetRightNode(i,j)
                curr_up = self.GetUpNode(i,j)
                curr_down = self.GetDownNode(i,j)

                if curr.type[0] == BlockType.EMPTY or curr.type[0] == BlockType.ME:
                    if curr_down and curr_down.type[0] == BlockType.NON_BREAKABLE or curr_down.type[0] == BlockType.BREAKABLE or curr_down.type[0] == BlockType.LADDER:
                        if curr_left and self.CanMoveToNode(curr_left):
                            joint = Joint((i-1, j))
                            curr.addJoint(joint)

                        if curr_right and self.CanMoveToNode(curr_right):
                            joint = Joint((i+1, j))
                            curr.addJoint(joint)

                    if curr_down and self.CanMoveToNode(curr_down):
                        joint = Joint((i, j+1))
                        curr.addJoint(joint)

                if curr.type[0] == BlockType.LADDER or curr.type[0] == BlockType.PIPE:
                    if curr_left and self.CanMoveToNode(curr_left):
                        joint = Joint((i-1, j))
                        curr.addJoint(joint)

                    if curr_right and self.CanMoveToNode(curr_right):
                        joint = Joint((i+1, j))
                        curr.addJoint(joint)

                    if curr.type[0] == BlockType.LADDER:
                      if curr_up and (curr_up.type[0] == BlockType.LADDER or curr_up.type[0] == BlockType.EMPTY):
                          joint = Joint((i, j-1))
                          curr.addJoint(joint)


                    if curr_down and self.CanMoveToNode(curr_down):
                        joint = Joint((i, j+1))
                        curr.addJoint(joint)
                
                #debug
                if False:
                    joints_str = ""
                    for tm in curr.joints:
                        joints_str += "(%s)" % str(tm.point)
                    f.write("(%d, %d): %d [%s]\n " % (i,j, curr.type[0], joints_str))


    def GetLeftNode(self,i,j):
        res = None
        if i < 0 or i >= self.size:
            return res

        res = self.matrix[i-1][j]
        return res

    def GetRightNode(self,i,j):
        res = None
        if i < 0 or i >= self.size-1:
            return res
        res = self.matrix[i+1][j]
        return res

    def GetUpNode(self,i,j):
        res = None
        if j < 1 or j >= self.size:
            return res
        res = self.matrix[i][j-1]
        return res
    
    def GetDownNode(self,i,j):
        res = None
        if j < 0 or j >= self.size-1:
            return res
        res = self.matrix[i][j+1]
        return res

    def CanMoveToNode(self, node):
        res = False
        typ = node.type[0]    
        if typ == BlockType.EMPTY or typ == BlockType.LADDER or typ == BlockType.GOLD or typ == BlockType.PIPE or typ == BlockType.ME:
            res = True
        return res

    def runWave(self, m, start, end):
        queue = []
        m[start[0]][start[1]] = 0
        queue.append(start)

        while queue:
            #print queue
            #f.write("%s" % m)
            elem = queue.pop(0)

            elem_age = m[elem[0]][elem[1]]
            if elem[0] == end[0] and elem[1] == end[1]:
                #we found it
                return True  
                

            joints = self.matrix[elem[0]][elem[1]].joints

            for joint in joints:
                j_point = joint.point
                if m[j_point[0]][j_point[1]] != None:
                    continue

                m[j_point[0]][j_point[1]] = elem_age + 1
                queue.append(j_point)

        return False

    def findNearestGold(self, start):
        m = [[None for x in range(self.size)] for x in range(self.size)]
        gold_pos = None
        queue = []
        m[start[0]][start[1]] = 0
        queue.append(start)

        while queue:
            #print queue
            #f.write("%s" % m)
            elem = queue.pop(0)

            elem_age = m[elem[0]][elem[1]]
            if self.matrix[elem[0]][elem[1]].type[0] == BlockType.GOLD:
                gold_pos = elem
                break   
                

            joints = self.matrix[elem[0]][elem[1]].joints

            for joint in joints:
                j_point = joint.point
                if m[j_point[0]][j_point[1]] != None:
                    continue

                m[j_point[0]][j_point[1]] = elem_age + 1
                queue.append(j_point)

        return gold_pos


    def getPrevAgeNeighbor(self, m, node):
        i = node[0]
        j = node[1]
        age = m[i][j]
        n_age = 0
        if i>0 and i<self.size:
            n_age = m[i-1][j]
            if n_age == age-1:
                return (i-1, j)

        if i>=0 and i<self.size-1:
            n_age = m[i+1][j]
            if n_age == age-1:
                return (i+1, j)

        if j>=0 and j<self.size-1:
            n_age = m[i][j+1]
            if n_age == age-1:
                return (i, j+1)

        if j>0 and j<self.size:
            n_age = m[i][j-1]
            if n_age == age-1:
                return (i, j-1)

        print "WARNING bad indexes"

    def waveBack(self, m, start, end):
        res = [end]
        prev = self.getPrevAgeNeighbor(m, end)
        res.append(prev)
        
        while (True):
            if prev[0] == start[0] and prev[1]== start[1]:
                break         
            prev = self.getPrevAgeNeighbor(m, prev)
            res.append(prev)

        return res





    def getRoute(self, start, end):
        path = []
        age_matrix = [[None for x in range(self.size)] for x in range(self.size)]
        res = self.runWave(age_matrix, start, end)
      
        if res:
            path = self.waveBack(age_matrix,start,end)

        return path
    
class Unit(object):

    def __init__(self):
        self.x = 50
        self.y = 50
        self.size = 20


class Solver():

    def __init__(self):
        print "Inited"
        self.goldPos = None
        self.route = None

    def solve(self, map):
        res = ''
        print "Me %s", map.me_point
        if True or self.goldPos is None:
            self.goldPos = map.findNearestGold(map.me_point)

        if not self.goldPos:
            print "WARNING - no gold found"
            return ""
        print "goldPos= %d %d " % (self.goldPos[0], self.goldPos[1])
        if True or self.route is None:
            self.route = map.getRoute(map.me_point, self.goldPos)
            self.route.reverse()
            self.route.pop(0)
 
        print "path= %s" % self.route
        
        p = None
        if self.route:
            p = self.route.pop(0)
            dx = p[0] - map.me_point[0]
            dy = p[1] - map.me_point[1]

                       
            #print self.route[0], p 
            if dx == 1:
                res = 'RIGHT'
            elif dx == -1:
                res = 'LEFT'
            elif dy == -1:
                res = 'UP'
            else:
                res = 'DOWN'
        else:
            print "WARNING - NO ROUTE"
        
        print  "Out res: %s" % res
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
                if map.matrix[i][j] and map.matrix[i][j].type[1]:
                    pygame.draw.circle(
                        self.m_window, map.matrix[i][j].type[1], [10 * i, 10 * j], 5)
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
                map.createGraphInfo()
                turn = solver.solve(map)
                ws_client.AccessVarResult(False, turn)

            self.DrawMap(map)
