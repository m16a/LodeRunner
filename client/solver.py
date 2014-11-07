# -*- coding: utf-8 -*-
import time
import math
import config
import ws_client
import random
import copy

if config.g_useRender:
    import pygame


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
    TRAP = 9
    HOLE = 10

legend = {
    u' ': BlockType.EMPTY,

    u"☼": BlockType.NON_BREAKABLE,

    u'#': BlockType.BREAKABLE,
    u'.': BlockType.HOLE,
    u'*': BlockType.HOLE,

    u'1': BlockType.TRAP,
    u'2': BlockType.TRAP,
    u'3': BlockType.TRAP,
    u'4': BlockType.TRAP,
    

    u'H': BlockType.LADDER,
    u'$': BlockType.GOLD,
    
    u'Ѡ': BlockType.ME,
    u'Я': BlockType.ME,
    u'R': BlockType.ME,
    u'Y': BlockType.ME,
    u'◄': BlockType.ME,
    u'►': BlockType.ME,
    u']': BlockType.ME,
    u'[': BlockType.ME,
    u'}': BlockType.ME,
    u'{': BlockType.ME,

    u'Q': BlockType.AI,
    u'«': BlockType.AI,
    u'»': BlockType.AI,
    u'<': BlockType.AI,
    u'>': BlockType.AI,
    u'X': BlockType.NON_BREAKABLE,

    u'~': BlockType.PIPE,

    u'Z': BlockType.ENEMY,
    u')': BlockType.ENEMY,
    u'(': BlockType.ENEMY,
    u'U': BlockType.ENEMY,
    u'Э': BlockType.ENEMY,
    u'Є': BlockType.ENEMY,

}


class Joint():
    def __init__(self, point):
        self.point = point
        self.val = None


class Node():
    def __init__(self, block_type):
        self.type = block_type
        self.joints = []
        self.draw_cell = None

    def add_joint(self, joint):
        self.joints.append(joint)

        
class Map():
    def __init__(self):
        self.matrix = None
        self.size = 0
        self.me_point = None
        self.look_direction = 'RIGHT'
        self.types_dict = {}

    def parse_msg(self, msg):
        msg_len = len(msg)
        sqrt_len = int(math.sqrt(msg_len))
        self.size = sqrt_len
        self.matrix = [
            [None for x in range(sqrt_len)] for x in range(sqrt_len)]

        for c in range(0, msg_len):
            if msg[c] not in legend.keys():
                print "WARNING: unknown symbol: "
                print msg[c].encode('utf-8')
                continue

            x = c % sqrt_len
            y = c // sqrt_len

            tmp = Node(legend[msg[c]])
            if tmp.type == BlockType.ME:
                self.me_point = (x, y)

            if msg[c] == u'Y':
                node = Node(legend[u'H'])
            elif msg[c] == u'{' or msg[c] == u'}':
                node = Node(legend[u'~'])
            else:
                node = Node(legend[msg[c]])

            node.draw_cell = Cell()
            self.matrix[x][y] = node

    def create_graph_info(self, alter=False):

        for i in range(self.size):
            for j in range(self.size):
                curr = self.matrix[i][j]
                if curr is None:
                    print "WARNING map no element"
                    continue

                curr_left = self.get_left_node(i, j)
                curr_right = self.get_right_node(i, j)
                curr_up = self.get_up_node(i, j)
                curr_down = self.get_down_node(i, j)

                if curr.type == BlockType.EMPTY or curr.type == BlockType.ME:
                    if curr_down and curr_down.type == BlockType.NON_BREAKABLE\
                            or curr_down.type == BlockType.BREAKABLE\
                            or curr_down.type == BlockType.LADDER\
                            or curr_down.type == BlockType.ENEMY\
                            or curr_down.type == BlockType.AI:
                        if curr_left and self.can_move_to_point((i-1, j)):
                            joint = Joint((i-1, j))
                            curr.add_joint(joint)

                        if curr_right and self.can_move_to_point((i+1, j)):
                            joint = Joint((i+1, j))
                            curr.add_joint(joint)

                    if curr_down:
                        if self.can_move_to_point((i, j+1)) or self.can_drill_and_move(i, j+1, alter):
                            joint = Joint((i, j+1))
                            curr.add_joint(joint)

                if curr.type == BlockType.LADDER or curr.type == BlockType.PIPE:
                    if curr_left and self.can_move_to_point((i-1, j)):
                        joint = Joint((i-1, j))
                        curr.add_joint(joint)

                    if curr_right and self.can_move_to_point((i+1, j)):
                        joint = Joint((i+1, j))
                        curr.add_joint(joint)

                    if curr.type == BlockType.LADDER:
                        if curr_up and curr_up.type \
                                in [BlockType.LADDER, BlockType.EMPTY, BlockType.GOLD, BlockType.PIPE]:
                            joint = Joint((i, j-1))
                            curr.add_joint(joint)

                    if curr_down and self.can_move_to_point((i, j+1)):
                        joint = Joint((i, j+1))
                        curr.add_joint(joint)

                if curr.type == BlockType.BREAKABLE and curr_down and self.can_move_to_point((i, j+1)):
                    joint = Joint((i, j+1))
                    curr.add_joint(joint)

    def dump_graph(self, file_name):
        with open(file_name, 'w') as f:
            graph_str = ''
            for i in range(self.size):
                for j in range(self.size):
                    curr = self.matrix[i][j]
                    joints_str = ""
                    if curr.joints:
                        joints_str = ''.join(["%s" % str(jn.point) for jn in curr.joints])
                    graph_str += "(%d, %d): %d [%s]\n " % (i, j, curr.type, joints_str)
            f.write(graph_str)

    def get_left_node(self, i, j):
        res = None
        if i < 0 or i >= self.size:
            return res

        res = self.matrix[i-1][j]
        return res

    def get_right_node(self, i, j):
        res = None
        if i < 0 or i >= self.size-1:
            return res
        res = self.matrix[i+1][j]
        return res

    def get_up_node(self, i, j):
        res = None
        if j < 1 or j >= self.size:
            return res
        res = self.matrix[i][j-1]
        return res
    
    def get_down_node(self, i, j):
        res = None
        if j < 0 or j >= self.size-1:
            return res
        res = self.matrix[i][j+1]
        return res

    def get_neighbor_points(self, point):
        neighbors = []
        x_point = point[0]
        y_point = point[1]
        if self.get_left_node(x_point, y_point):
            neighbors.append((x_point - 1, y_point))
        if self.get_right_node(x_point, y_point):
            neighbors.append((x_point + 1, y_point))
        if self.get_up_node(x_point, y_point):
            neighbors.append((x_point, y_point - 1))
        if self.get_down_node(x_point, y_point):
            neighbors.append((x_point, y_point + 1))
        return neighbors

    def get_neighbor_nodes_types(self, point):
        return [self.get_type(p) for p in self.get_neighbor_points(point)]

    def can_move_to_point(self, end_point):
        end_type = self.get_type(end_point)

        if end_type in [BlockType.TRAP, BlockType.AI, BlockType.ENEMY, BlockType.NON_BREAKABLE, BlockType.BREAKABLE]:
            return False

        if end_type in [BlockType.EMPTY, BlockType.LADDER, BlockType.PIPE, BlockType.GOLD, BlockType.ME]:
            return True

        if end_type == BlockType.HOLE and self.can_move_to_point((end_point[0], end_point[1]+1)):
            return True

        return False

    def can_ai_get_me(self, point):
        return BlockType.AI in self.get_neighbor_nodes_types(point)

    def can_drill_and_move(self, x, y, alter):
        return self.matrix[x][y].type == BlockType.BREAKABLE and self.can_move_to_point((x, y+1))\
            and not self.me_point == (x, y-1) and not alter

    def run_wave(self, m, start, end):
        queue = []
        m[start[0]][start[1]] = 0
        queue.append(start)

        while queue:
            elem = queue.pop(0)

            elem_age = m[elem[0]][elem[1]]
            if elem[0] == end[0] and elem[1] == end[1]:
                #we found it
                return True  
                
            joints = self.matrix[elem[0]][elem[1]].joints

            for joint in joints:
                j_point = joint.point
                if m[j_point[0]][j_point[1]] is not None:
                    continue

                m[j_point[0]][j_point[1]] = elem_age + 1
                queue.append(j_point)

        return False

    def find_nearest_type(self, start, block_type):
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
            if self.matrix[elem[0]][elem[1]].type == block_type:
                gold_pos = elem
                break

            joints = self.matrix[elem[0]][elem[1]].joints

            for joint in joints:
                j_point = joint.point
                if m[j_point[0]][j_point[1]] is not None:
                    continue

                m[j_point[0]][j_point[1]] = elem_age + 1
                queue.append(j_point)

        return gold_pos

    def find_gold_candidates(self, start):
        m = [[None for x in range(self.size)] for x in range(self.size)]
        gold_pos = None
        queue = []
        m[start[0]][start[1]] = 0
        queue.append(start)
        gold_candidates = []

        while queue and len(gold_candidates) < 3:

            elem = queue.pop(0)

            elem_age = m[elem[0]][elem[1]]
            if self.matrix[elem[0]][elem[1]].type == BlockType.GOLD:
                gold_candidates.append(elem)

            joints = self.matrix[elem[0]][elem[1]].joints

            for joint in joints:
                j_point = joint.point
                if m[j_point[0]][j_point[1]] is not None:
                    continue

                m[j_point[0]][j_point[1]] = elem_age + 1
                queue.append(j_point)

        return gold_candidates

    def get_prev_age_neighbor(self, m, node):
        i = node[0]
        j = node[1]
        age = m[i][j]
        n_age = 0
        if 0 < i < self.size:
            n_age = m[i-1][j]
            if n_age == age-1 and node in [joint.point for joint in self.matrix[i-1][j].joints]:
                return i-1, j

        if 0 <= i < self.size-1:
            n_age = m[i+1][j]
            if n_age == age-1 and node in [joint.point for joint in self.matrix[i+1][j].joints]:
                return i+1, j

        if 0 <= j < self.size-1:
            n_age = m[i][j+1]
            if n_age == age-1 and node in [joint.point for joint in self.matrix[i][j+1].joints]:
                return i, j+1

        if 0 < j < self.size:
            n_age = m[i][j-1]
            if n_age == age-1 and node in [joint.point for joint in self.matrix[i][j-1].joints]:
                return i, j-1

        print "WARNING bad indexes"

    def wave_back(self, m, start, end):
        res = [end]
        prev = self.get_prev_age_neighbor(m, end)
        res.append(prev)
        
        while True:
            if prev[0] == start[0] and prev[1] == start[1]:
                break         
            prev = self.get_prev_age_neighbor(m, prev)
            res.append(prev)

        return res

    def get_route(self, start, end):
        path = []
        age_matrix = [[None for x in range(self.size)] for x in range(self.size)]
        res = self.run_wave(age_matrix, start, end)
      
        if res:
            path = self.wave_back(age_matrix, start, end)
            path.reverse()

        return path[1:]

    def can_escape(self):
        pit_position = self.get_pit_position(self.look_direction)
        if pit_position:
            print "pit_position: %s:%s" % (pit_position[0], pit_position[1])
            if self.can_drill_point(pit_position) and self.can_move_to_point((pit_position[0], pit_position[1] + 1)):
                return True
        return False

    def get_pit_position(self, direction):
        if direction == 'RIGHT':
            return self.me_point[0] + 1, self.me_point[1] + 1
        else:
            return self.me_point[0] - 1, self.me_point[1] + 1

    def can_drill_point(self, point):
        return self.get_type(point) == BlockType.BREAKABLE and self.get_type((point[0], point[1]-1)) != BlockType.LADDER

    def get_type(self, point):
        return self.matrix[point[0]][point[1]].type

    def is_ai_next_to_me(self, point):
        return abs(point[0] - self.me_point[0]) == 1

    def get_node(self, point):
        return self.matrix[point[0]][point[1]]


class Unit(object):

    def __init__(self):
        self.x = 50
        self.y = 50
        self.size = 20


class Solver():

    def __init__(self):
        self.goldPos = None
        self.route = None
        self.queue = []
        self.me_point = None
        self.curr_node = None
        self.gold_candidates = []
        self.alter_gold_candidates = []
        self.game_map = None
        self.surround_points = None
        self.is_in_danger = None
        self.previous_point = None
        self.alter_map = None

    def solve(self, game_map):
        self.turn_related_assignments(game_map)

        if self.queue:
            print "self.queue: %s" % self.queue
            return self.queue.pop(0)

        if self.is_in_danger:
            action = self.get_action_when_in_danger()
            if action:
                return action

        if not self.goldPos:
            return self.get_no_gold_action()

        self.route = self.game_map.get_route(self.me_point, self.goldPos)
        print "path= %s" % self.route

        if self.route:
            next_point = self.route[0]

            if self.game_map.can_ai_get_me(next_point):
                print "AI can get me!!! Get Back!!!"
                return get_direction(self.me_point, self.previous_point)

            if self.is_gold_under_me(game_map):
                print "Gold is under me"
                turn = get_direction(self.me_point, next_point)
                turn_back = get_opposite_direction(turn)
                self.queue.append('%s,%s' % ('ACT', turn_back))
                self.queue.append(turn_back)
                return turn

            if self.should_drill():
                direction = get_direction(self.me_point, self.route[0])
                self.queue.append(direction)
                return '%s,%s' % ('ACT', direction)

            else:
                return get_direction(game_map.me_point, self.route.pop(0))

        else:
            print "WARNING - no route found"
            return ''

    def turn_related_assignments(self, game_map):
        self.game_map = game_map
        self.alter_map = copy.deepcopy(self.game_map)
        self.alter_map.create_graph_info(True)
        self.alter_map.dump_graph('dump_Grapth_Alter.txt') # Debug
        self.me_point = self.game_map.me_point
        self.curr_node = self.game_map.matrix[self.me_point[0]][self.me_point[1]]
        self.surround_points = self.get_surround_points()
        self.goldPos = self.game_map.find_nearest_type(self.me_point, BlockType.GOLD)
        self.gold_candidates = self.game_map.find_gold_candidates(self.me_point)
        self.alter_gold_candidates = self.alter_map.find_gold_candidates(self.me_point)
        self.is_in_danger = self.is_player_in_danger()
        print "Me %s: %s" % ([self.me_point], self.curr_node.type)
        print "gold_candidates: %s" % self.gold_candidates
        print "alter_gold_candidates: %s" % list(set(self.alter_gold_candidates) - set(self.gold_candidates))

    def is_player_in_danger(self):
        if BlockType.AI in [self.game_map.get_type(p) for p in self.surround_points]:
            return True
        return False

    def get_action_when_in_danger(self):
        print "Player is in danger !!!"
        nearest_ai_point = self.get_nearest_ai_point()
        print "Nearest AI point: %s" % [nearest_ai_point]
        if not self.game_map.is_ai_next_to_me(nearest_ai_point):
            nearest_ai_direction = get_direction(self.me_point, nearest_ai_point)
            drill_point = get_drill_point(self.me_point, get_multiplier(nearest_ai_direction))
            print "drill_point: %s, type: %s" % (drill_point, self.game_map.get_type(drill_point))
            print "can drill: %s" % self.game_map.can_drill_point(drill_point)
            if self.game_map.can_drill_point(drill_point) and self.curr_node.type != BlockType.PIPE:
                print "Making a trap ...."
                return '%s,%s' % ('ACT', nearest_ai_direction)
            else:
                print "Escaping"
                direction = get_direction(nearest_ai_point, self.me_point)
                if self.can_move_in_direction(direction):
                    return get_direction(nearest_ai_point, self.me_point)
                else:
                    return None

    def get_nearest_ai_point(self):
        for point in self.surround_points:
            if self.game_map.get_type(point) == BlockType.AI:
                return point

    def can_move_in_direction(self, direction):
        point = self.me_point[0] + get_multiplier(direction), self.me_point[1]
        return self.game_map.can_move_to_point(point)

    def should_drill(self):
        if len(self.route) > 1:
            breakable_next_to_next = self.game_map.get_type(self.route[1]) == BlockType.BREAKABLE
            x_point_of_block_is_next_to_me = abs(self.me_point[0] - self.route[1][0]) == 1
            ai_cant_get_me = BlockType.AI not in self.game_map.get_neighbor_nodes_types(self.me_point)
            return breakable_next_to_next and x_point_of_block_is_next_to_me and ai_cant_get_me

    def get_no_gold_action(self):
        print "WARNING - no gold found"
        if self.game_map.can_escape():
            print "Escape plan"
            self.queue.append(self.game_map.look_direction)
            return 'ACT'
        else:
            joints = self.game_map.get_node(self.me_point).joints
            if joints:
                index = random.randint(0, len(joints) - 1)
                return get_direction(self.me_point, joints[index].point)
            return ''

    def is_gold_under_me(self, game_map):
        if game_map.me_point[0] == self.goldPos[0]:
            if game_map.get_down_node(self.me_point[0], self.me_point[1]).type == BlockType.BREAKABLE\
                    and self.curr_node.type != BlockType.LADDER:
                start_point = (self.me_point[0], self.me_point[1] + 1)
                route = game_map.get_route(start_point, self.goldPos)
                print "route to gold under me: %s" % route
                route_node_types = [game_map.matrix[point[0]][point[1]].type for point in route][:-1]
                if route_node_types:
                    if route_node_types.count(route_node_types[0]) == len(route_node_types):
                        return True
        return False

    def get_surround_points(self):
        surround_points = []
        multipliers = [-1, 1]
        for i in range(1, 5):
            for multiplier in multipliers:
                x_position = self.me_point[0] + multiplier * i

                if x_position < self.game_map.size:
                    surround_points.append((x_position, self.me_point[1]))
        return surround_points

    def remember_previous_point(self):
        self.previous_point = self.me_point[0], self.me_point[1]


def get_opposite_direction(direction):
    if direction == 'RIGHT':
        return 'LEFT'
    else:
        return 'RIGHT'


def get_multiplier(direction):
    if direction == 'RIGHT':
        return 1
    else:
        return -1


def get_drill_point(point, multiplier):
    return point[0] + 1 * multiplier, point[1] + 1


def get_direction(me_point, target_point):
    dx = target_point[0] - me_point[0]
    dy = target_point[1] - me_point[1]

    if dx > 0:
        return 'RIGHT'
    elif dx < 0:
        return 'LEFT'
    elif dy < 0:
        return 'UP'
    elif dy > 0:
        return 'DOWN'
    else:
        return ''

class Cell():
    def __init__(self):
        self.bgr_color = None
        self.size = 16
        self.type = None


class Render():

    def __init__(self):
        if config.g_useRender: 
            pygame.init()
            self.m_window = pygame.display.set_mode((1000, 1000))
            print "Render inited"
        else:
            print "Render is NOT used"
        
    def start(self):
        msg = None
        game_map = Map()
        self.solver = Solver()
        while True:
            time.sleep(0.1)
            if config.g_useRender:
                for event in pygame.event.get():
                    NOP = 1
            if config.newMSG:
                msg = ws_client.AccessVarMSG(True, None)
                game_map.parse_msg(msg)
                game_map.create_graph_info()
                game_map.dump_graph('dump_Grapth.txt') # Debug
                turn = self.solver.solve(game_map)
                self.solver.remember_previous_point()
                print "turn: %s" % turn
                if turn in ['LEFT', 'RIGHT']:
                    if game_map.look_direction != turn:
                        game_map.look_direction = turn
                print "Look direction: %s" % game_map.look_direction
                ws_client.AccessVarResult(False, turn)
                if config.g_useRender:
                    self.DrawMap(game_map)

    def DrawMap(self, map):
        if not config.g_useRender:
            return

        if not map.matrix:
            return

        self.m_window.fill((224, 224, 224))

        for i in range(map.size):
            for j in range(map.size):
                if map.matrix[i][j]:
                    node = map.matrix[i][j]
                    self.drawNode(node,i,j)
                    

        # pygame.draw.circle(self.m_window, BLUE, [100, 100], 20)
        pygame.display.flip()

    def drawNode(self, node,i,j):
        size = node.draw_cell.size

        bgrd_color = None

        if node.type == BlockType.NON_BREAKABLE:
            bgrd_color = (64, 64, 64)
        elif node.type == BlockType.BREAKABLE:
            bgrd_color = (255, 204, 153)
        elif node.type == BlockType.ME:
            bgrd_color = (0, 255, 0)
        elif node.type == BlockType.AI:
            bgrd_color = (255, 0, 0)
        elif node.type == BlockType.GOLD:
            bgrd_color = (255, 255, 0)

        if bgrd_color:
            pygame.draw.rect(
                        self.m_window, bgrd_color, (size * i, size * j, size+1, size+1), 0)

        #path
        center = (int(size * (i+0.5)), int(size * (j+0.5)))
        if (i,j) in self.solver.route:
            pygame.draw.circle(self.m_window, (51,153,255), center, 5)


        #grid
        pygame.draw.rect(
                        self.m_window, (128, 128, 128), (size * i, size * j, size+1, size+1), 1)
        
        #pygame.draw.line(self.m_window, (128, 0, 0), center, end, 1)
        for joint in node.joints:
            delta = (joint.point[0]-i, joint.point[1]-j)
            end = (center[0] + delta[0] * size / 2, center[1] + delta[1] * size / 2)
            pygame.draw.line(self.m_window, (128, 0, 0), center, end, 1)

"""
TODO:
1. improve should_drill (alternative grapth where we can't move through breakable)
2. better handling no gold found
3. fix gold is under me
4. improve logistics
"""