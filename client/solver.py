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
    ME_on_Ladder = 11
    AI_on_Ladder = 12
    ENEMY_on_Ladder = 13
    ME_on_Pipe = 14
    AI_on_Pipe = 15
    ENEMY_on_Pipe = 16

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
    u'Y': BlockType.ME_on_Ladder,
    u'◄': BlockType.ME,
    u'►': BlockType.ME,
    u']': BlockType.ME,
    u'[': BlockType.ME,
    u'}': BlockType.ME_on_Pipe,
    u'{': BlockType.ME_on_Pipe,

    u'Q': BlockType.AI_on_Ladder,
    u'«': BlockType.AI,
    u'»': BlockType.AI,
    u'<': BlockType.AI_on_Pipe,
    u'>': BlockType.AI_on_Pipe,
    u'X': BlockType.NON_BREAKABLE,

    u'~': BlockType.PIPE,

    u'Z': BlockType.ENEMY,
    u')': BlockType.ENEMY,
    u'(': BlockType.ENEMY,
    u'U': BlockType.ENEMY_on_Ladder,
    u'Э': BlockType.ENEMY_on_Pipe,
    u'Є': BlockType.ENEMY_on_Pipe,

}

class Person():
    ME = 1
    AI = 2
    ENEMY = 3

class Joint():
    def __init__(self, point):
        self.point = point
        self.val = None


class Node():
    def __init__(self, block_type):
        self.type = block_type
        self.joints = []
        self.draw_cell = None
        self.person = None

    def add_joint(self, joint):
        self.joints.append(joint)

        
class Map():
    def __init__(self):
        self.matrix = None
        self.size = 0
        self.me_point = None
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

            
            new_legend = legend[msg[c]]
            tmp = Node(new_legend)
            person = None

            if tmp.type in [BlockType.ME_on_Ladder, BlockType.ME, BlockType.ME_on_Pipe]:
                self.me_point = (x, y)
                #print "!!!!!!!accept Person"
                person = Person.ME
            elif tmp.type in [BlockType.AI_on_Ladder, BlockType.AI, BlockType.AI_on_Pipe]:
                person = Person.AI
            elif tmp.type in [BlockType.ENEMY_on_Ladder, BlockType.ENEMY, BlockType.ENEMY_on_Pipe]:
                person = Person.ENEMY


            if tmp.type in [BlockType.ME_on_Ladder, BlockType.AI_on_Ladder, BlockType.ENEMY_on_Ladder]:
                new_legend = BlockType.LADDER
            elif tmp.type in [BlockType.ME_on_Pipe, BlockType.AI_on_Pipe, BlockType.ENEMY_on_Pipe]:
                new_legend = BlockType.PIPE
            elif tmp.type in [BlockType.ME, BlockType.AI, BlockType.ENEMY]:
                new_legend = BlockType.EMPTY

            node = Node(new_legend)
            node.person = person
            node.draw_cell = Cell()
            self.matrix[x][y] = node

    def update_graph_info(self):
        for i in range(self.size):
            for j in range(self.size):
                #disable graph joints near close enemy
                if self.matrix[i][j].person == Person.ENEMY or self.matrix[i][j].person == Person.AI:
                    if abs(i - self.me_point[0]) + abs(j - self.me_point[1]) < 10:
                        neighbors = self.get_neighbor_points((i,j))

                        for n in neighbors:
                            node = self.matrix[n[0]][n[1]]
                            node.joints = [item for item in node.joints if item.point != (i,j)]
                        
    def create_graph_info(self):
        for i in range(self.size):
            for j in range(self.size):
                curr_point = i, j
                curr_node = self.get_node(curr_point)
                if self.get_node(curr_point) is None:
                    print "WARNING map no element"
                    continue

                if self.can_move_up(curr_point):
                    joint = Joint(get_up_point(curr_point))
                    curr_node.add_joint(joint)

                if self.can_move_down(curr_point):
                    joint = Joint(get_down_point(curr_point))
                    curr_node.add_joint(joint)

                elif self.can_drill_and_move(curr_point):
                    joint = Joint(get_down_point(curr_point))
                    curr_node.add_joint(joint)

                if self.can_move_right(curr_point):
                    joint = Joint(get_right_point(curr_point))
                    curr_node.add_joint(joint)

                if self.can_move_left(curr_point):
                    joint = Joint(get_left_point(curr_point))
                    curr_node.add_joint(joint)

    def can_move_up(self, point):
        return self.get_type(point) in [BlockType.LADDER,
                                        BlockType.ME_on_Ladder, BlockType.AI_on_Ladder, BlockType.ENEMY_on_Ladder]\
            and self.can_move_to_point(get_up_point(point))

    def can_move_down(self, point):
        return self.get_type(point) in [BlockType.LADDER, BlockType.PIPE, BlockType.ME, BlockType.EMPTY, BlockType.GOLD,
                                        BlockType.HOLE, BlockType.BREAKABLE, BlockType.AI,
                                        BlockType.ME_on_Ladder, BlockType.AI_on_Ladder, BlockType.ENEMY_on_Ladder]\
            and self.can_move_to_point(get_down_point(point))#\
            #or self.get_type(get_down_point(point)) == BlockType.BREAKABLE and\
            #self.get_type(get_down_point(get_down_point(point))) not in [BlockType.BREAKABLE, BlockType.NON_BREAKABLE]

    def can_move_right(self, point):
        if self.can_move_to_point(get_right_point(point)):
            return (self.get_type(point) in [BlockType.EMPTY, BlockType.ME, BlockType.PIPE, BlockType.LADDER,
                                             BlockType.GOLD, BlockType.ME_on_Ladder, BlockType.AI_on_Ladder,
                                             BlockType.ENEMY_on_Ladder] and self.can_stand_on(get_down_point(point)))\
                or self.get_type(point) in [BlockType.LADDER, BlockType.PIPE]

    def can_move_left(self, point):
        if self.can_move_to_point(get_left_point(point)):
            return (self.get_type(point) in [BlockType.EMPTY, BlockType.ME, BlockType.PIPE, BlockType.LADDER,
                                             BlockType.GOLD, BlockType.ME_on_Ladder, BlockType.AI_on_Ladder,
                                             BlockType.ENEMY_on_Ladder] and self.can_stand_on(get_down_point(point)))\
                or self.get_type(point) in [BlockType.LADDER, BlockType.PIPE]  

    def is_pipe(self, point):
        return self.get_type(point) in [BlockType.PIPE, BlockType.ME_on_Pipe,
                                        BlockType.AI_on_Pipe, BlockType.ENEMY_on_Pipe]

    def can_drill_and_move(self, point):
        return point[1]+2 < self.size and self.get_type(get_down_point(point)) == BlockType.BREAKABLE\
            and not self.get_type(point) in [BlockType.LADDER, BlockType.ENEMY_on_Ladder, BlockType.AI_on_Ladder] \
            and self.can_move_to_point(get_down_point((point[0], point[1]+1)))\
            and (self.can_move_to_point(get_left_point(point)) or self.can_move_to_point(get_right_point(point))) 

    def can_move_to_point(self, end_point):
        if end_point[0] < self.size and end_point[1] < self.size:
            end_type = self.get_type(end_point)

            if end_type in [BlockType.TRAP, BlockType.AI, BlockType.ENEMY, BlockType.NON_BREAKABLE, BlockType.BREAKABLE]:
                return False
            if end_type in [BlockType.EMPTY, BlockType.LADDER, BlockType.PIPE, BlockType.GOLD, BlockType.ME,
                            BlockType.ME_on_Ladder]:
                return True
            if end_type == BlockType.HOLE and self.can_move_to_point((end_point[0], end_point[1]+1)):
                return True
        return False

    def can_stand_on(self, point):
        return self.get_type(point) in [BlockType.BREAKABLE, BlockType.NON_BREAKABLE, BlockType.LADDER, BlockType.ENEMY,
                                        BlockType.ME_on_Ladder]

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

    def can_ai_get_me(self, point):
        return BlockType.AI in self.get_neighbor_nodes_types(point)

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

    def get_scan_dict(self, start):
        m = [[None for x in range(self.size)] for x in range(self.size)]
        queue = []
        m[start[0]][start[1]] = 0
        queue.append(start)
        scan_dict = dict()
        scan_dict['gold'] = [] # (point, dist, priority)


        while queue:
            elem = queue.pop(0)
            elem_age = m[elem[0]][elem[1]]
            if self.matrix[elem[0]][elem[1]].type == BlockType.GOLD:
                t = [elem, elem_age, 0]
                scan_dict['gold'].append(t)

            joints = self.matrix[elem[0]][elem[1]].joints

            for joint in joints:
                j_point = joint.point
                if m[j_point[0]][j_point[1]] is not None:
                    continue

                m[j_point[0]][j_point[1]] = elem_age + 1
                queue.append(j_point)
        
        for g in scan_dict['gold']:
            for i in range(self.size):
                for j in range(self.size):
                    m[i][j] = None
            start_point = g[0]
            queue = []
            m[start_point[0]][start_point[1]] = 0
            queue.append(start_point)

            while queue:
                elem = queue.pop(0)
                elem_age = m[elem[0]][elem[1]]
                if elem_age > 10:
                    break

                if self.matrix[elem[0]][elem[1]].type == BlockType.GOLD:
                    g[2] += 1
                
                joints = self.matrix[elem[0]][elem[1]].joints

                for joint in joints:
                    j_point = joint.point
                    if m[j_point[0]][j_point[1]] is not None:
                        continue

                    m[j_point[0]][j_point[1]] = elem_age + 1
                    queue.append(j_point)

        return scan_dict

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

    def get_pit_position(self, direction):
        return self.me_point[0] + get_multiplier(direction), self.me_point[1] + 1

    def can_drill_point(self, point):
        return self.get_type(point) == BlockType.BREAKABLE and self.get_type((point[0], point[1]-1)) != BlockType.LADDER

    def get_type(self, point):
        return self.matrix[point[0]][point[1]].type

    def is_ai_next_to_me(self, point):
        return abs(point[0] - self.me_point[0]) == 1

    def get_node(self, point):
        return self.matrix[point[0]][point[1]]


def get_up_point(point):
    return point[0], point[1] - 1


def get_down_point(point):
    return point[0], point[1] + 1


def get_left_point(point):
    return point[0] - 1, point[1]


def get_right_point(point):
    return point[0] + 1, point[1]


class Unit(object):

    def __init__(self):
        self.x = 50
        self.y = 50
        self.size = 20


class Solver():

    def __init__(self):
        self.route = None
        self.queue = []
        self.me_point = None
        self.curr_node = None
        self.gold_candidates = []
        self.game_map = None
        self.surround_points = None
        self.is_in_danger = None
        self.previous_point = None
        self.ai_points = None
        self.enemy_points = None
        self.drill_route = None

    def solve(self, game_map):
        self.turn_related_assignments(game_map)

        if self.queue:
            print "self.queue: %s" % self.queue
            return self.queue.pop(0)

        if self.is_in_danger:
            action = self.get_action_when_in_danger()
            if action:
                return action

        if not self.gold_candidates:
            #self.goldPos = self.get_random_gold()
            return ''

        self.route = self.game_map.get_route(self.me_point, self.gold_candidates[0][0])
        #print "path= %s" % self.route

        if self.route:
            next_point = self.route[0]

            if self.game_map.can_ai_get_me(next_point):
                print "AI can get me!!! Get Back!!!"
                if self.previous_point:
                    return get_direction(self.me_point, self.previous_point)

            res = self.should_drill_down()

            if res:
                self.queue = res
                return self.queue.pop(0)
            
            if self.is_gold_under_me():
                return self.get_action_when_gold_is_under_me()
            
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
        self.me_point = self.game_map.me_point
        self.curr_node = self.game_map.matrix[self.me_point[0]][self.me_point[1]]
        self.surround_points = self.get_surround_points()

        scan_dict = self.game_map.get_scan_dict(self.me_point)

        self.gold_candidates = scan_dict['gold']

        #(point, dist, gold_near)
        A = 1
        C = 150
        B = 12
        self.gold_candidates.sort(key=lambda x: A * (C - x[1]) + B * x[2], reverse=True)

        self.is_in_danger = self.is_player_in_danger()
        print "Me %s: %s" % ([self.me_point], self.curr_node.type)
        print "gold_candidates: %s" % self.gold_candidates
        print "AI points: %s" % self.ai_points
        print "ENEMY points: %s" % self.enemy_points

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
            if self.game_map.get_type(point) in [BlockType.AI, BlockType.AI_on_Ladder, BlockType.AI_on_Pipe]:
                return point

    def can_move_in_direction(self, direction):
        point = self.me_point[0] + get_multiplier(direction), self.me_point[1]
        return self.game_map.can_move_to_point(point)

    def should_drill_down(self):
        tmp_route = None
        if len(self.route) > 1:
            if self.game_map.get_type(self.route[0]) == BlockType.BREAKABLE:
                print "Need to dig down"
                if self.game_map.can_move_to_point(get_left_point(self.me_point)) \
                and (self.game_map.can_stand_on(get_left_point(self.route[0])) or  BlockType.PIPE == self.game_map.get_type(get_left_point(self.route[0]))):
                    #self.route.insert(0, get_left_point(self.me_point))
                    tmp_route = ['LEFT', 'ACT, RIGHT', 'RIGHT']
                elif self.game_map.can_move_to_point(get_right_point(self.me_point))\
                and (self.game_map.can_stand_on(get_right_point(self.route[0])) or  BlockType.PIPE == self.game_map.get_type(get_right_point(self.route[0]))):
                    #self.route.insert(0, get_right_point(self.me_point))
                    tmp_route = ['RIGHT', 'ACT, LEFT', 'LEFT']
                else:
                    print "WARNING - bad drilling"
        return tmp_route

    def should_drill(self):
        if len(self.route) > 1:
            breakable_next_to_next = self.game_map.get_type(self.route[1]) == BlockType.BREAKABLE
            
            x_point_of_block_is_next_to_me = abs(self.me_point[0] - self.route[1][0]) == 1
            ai_cant_get_me = BlockType.AI not in self.game_map.get_neighbor_nodes_types(self.me_point)
            return breakable_next_to_next and x_point_of_block_is_next_to_me and ai_cant_get_me


    def is_gold_under_me(self):
        if self.me_point[0] == self.gold_candidates[0][0][0]:
            if self.game_map.get_type((self.me_point[0], self.me_point[1]+2)) == BlockType.BREAKABLE\
                    and self.curr_node.type != BlockType.LADDER:
                start_point = (self.me_point[0], self.me_point[1] + 2)
                route = self.game_map.get_route(start_point, self.gold_candidates[0][0])
                route_node_types = [self.game_map.get_type(point) for point in route][:-1]
                if route_node_types:
                    if route_node_types.count(route_node_types[0]) == len(route_node_types):
                        return True
        return False

    def get_action_when_gold_is_under_me(self):
        print "Gold is under me"
        for multiplier in [-1, 1]:
            if self.game_map.can_move_to_point((self.me_point[0]+multiplier, self.me_point[1]+1))\
                    and self.game_map.can_stand_on((self.me_point[0]+multiplier, self.me_point[1]+2)):
                self.queue.append(get_direction(self.me_point, (self.me_point[0] + multiplier, self.me_point[1])))
            return ''

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
                game_map.update_graph_info()
                #game_map.dump_graph('dump_Grapth.txt') # Debug
                turn = self.solver.solve(game_map)
                self.solver.remember_previous_point()
                print "turn: %s" % turn
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

        if node.person:
            if node.person == Person.ME:
                bgrd_color = (0, 255, 0)
            elif node.person == Person.AI:
                bgrd_color = (255, 0, 0)
            elif node.person == Person.ENEMY:
                bgrd_color = (255, 150, 0)
        else:

            if node.type == BlockType.NON_BREAKABLE:
                bgrd_color = (64, 64, 64)
            elif node.type == BlockType.BREAKABLE:
                bgrd_color = (255, 204, 153)
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