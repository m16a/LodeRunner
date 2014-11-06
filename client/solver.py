# -*- coding: utf-8 -*-
import time
import math
import config
import ws_client
import random

g_useRender = False


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

legend = {
    u' ': BlockType.EMPTY,

    u"☼": BlockType.NON_BREAKABLE,

    u'#': BlockType.BREAKABLE,
    u'.': BlockType.TRAP,
    u'*': BlockType.TRAP,
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
    u'X': BlockType.AI,

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

            self.matrix[x][y] = node

    def create_graph_info(self):
        with open('dump_Graph.txt', 'w') as f:
            for i in range(self.size):
                for j in range(self.size):
                    curr = self.matrix[i][j]
                    if curr is None:
                        print "WARNING map no element"
                        continue

                    #create joints

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
                            if curr_left and self.can_move_to_node(curr_left, i, j):
                                joint = Joint((i-1, j))
                                curr.add_joint(joint)

                            if curr_right and self.can_move_to_node(curr_right, i, j):
                                joint = Joint((i+1, j))
                                curr.add_joint(joint)

                        if curr_down:
                            if self.can_move_to_node(curr_down, i, j)\
                                    or (self.can_drill_and_move(i, j+1) and not self.me_point == (i, j)):
                                joint = Joint((i, j+1))
                                curr.add_joint(joint)

                    if curr.type == BlockType.LADDER or curr.type == BlockType.PIPE:
                        if curr_left and self.can_move_to_node(curr_left, i, j):
                            joint = Joint((i-1, j))
                            curr.add_joint(joint)

                        if curr_right and self.can_move_to_node(curr_right, i, j):
                            joint = Joint((i+1, j))
                            curr.add_joint(joint)

                        if curr.type == BlockType.LADDER:
                            if curr_up and (curr_up.type == BlockType.LADDER or curr_up.type == BlockType.EMPTY
                                            or curr_up.type == BlockType.GOLD or curr_up.type == BlockType.PIPE):
                                joint = Joint((i, j-1))
                                curr.add_joint(joint)

                        if curr_down and self.can_move_to_node(curr_down, i, j):
                            joint = Joint((i, j+1))
                            curr.add_joint(joint)

                    if curr.type == BlockType.BREAKABLE and curr_down and self.can_move_to_node(curr_down, i, j):
                        joint = Joint((i, j+1))
                        curr.add_joint(joint)

            #debug dump Graph
            if True:
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

    def can_move_to_node(self, node, x_point, y_point):
        typ = node.type
        down_node_typ = self.matrix[x_point][y_point].type
        if down_node_typ == BlockType.TRAP:
            if typ == BlockType.EMPTY or typ == BlockType.GOLD:
                return False
        if typ == BlockType.EMPTY or typ == BlockType.LADDER or typ == BlockType.GOLD\
                or typ == BlockType.PIPE or typ == BlockType.ME:
            return True
        return False

    def can_ai_get_me(self, point):
        if BlockType.AI in [node.type for node in [self.matrix[p[0]][p[1]] for p in self.get_revertible_points(point)]]:
            return True
        return False

    def get_revertible_points(self, point):
        next_points = self.get_neighbor_points(point)
        return [p for p in next_points if point in [j.point for j in self.matrix[p[0]][p[1]].joints]]

    def can_drill_and_move(self, x, y):
        return self.matrix[x][y].type == BlockType.BREAKABLE and self.can_move_to_node(self.get_down_node(x, y), x, y)

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
        return self.matrix[point[0]][point[1]].type == BlockType.BREAKABLE\
            and self.matrix[point[0]][point[1] - 1].type != BlockType.LADDER

    def can_move_to_point(self, point):
        return self.can_move_to_node(self.matrix[point[0]][point[1]], point[0], point[1])

    def can_move_forward(self):
        return self.can_move_to_point(self.next_horizontal_point())

    def next_horizontal_point(self):
        if self.look_direction == 'RIGHT':
            return self.me_point[0] + 1, self.me_point[1]
        else:
            return self.me_point[0] - 1, self.me_point[1]

    def past_horizontal_point(self):
        if self.look_direction == 'RIGHT':
            return self.me_point[0] - 1, self.me_point[1]
        else:
            return self.me_point[0] + 1, self.me_point[1]

    def next_horizontal_node(self):
        next_point = self.next_horizontal_point()
        return self.matrix[next_point[0]][next_point[1]]

    def past_horizontal_node(self):
        past_point = self.past_horizontal_point()
        return self.matrix[past_point[0]][past_point[1]]

    def get_type(self, point):
        return self.matrix[point[0]][point[1]].type

    def is_ai_next_to_me(self, point):
        return abs(point[0] - self.me_point[0]) == 1


class Unit(object):

    def __init__(self):
        self.x = 50
        self.y = 50
        self.size = 20


class Solver():

    def __init__(self):
        print "Inited"
        self.goldPos = None
        self.ai_position = Node
        self.enemy_position = None
        self.route = None
        self.queue = []
        self.me_point = None
        self.curr_node = None

    def solve(self, game_map):
        self.me_point = game_map.me_point
        self.curr_node = game_map.matrix[self.me_point[0]][self.me_point[1]]
        print "Me %s: %s" % ([self.me_point], self.curr_node.type)


        surround_points = self.get_surround_points(game_map)
        if is_player_chased(surround_points, game_map):
            print "Player is being chased"
            nearest_ai_point = get_nearest_ai_point(surround_points, game_map)
            print "Nearest AI point: %s" % [nearest_ai_point]
            if not game_map.is_ai_next_to_me(nearest_ai_point):
                nearest_ai_direction = get_direction(self.me_point, get_nearest_ai_point(surround_points, game_map))
                drill_point = get_drill_point(self.me_point, get_multiplier(nearest_ai_direction))
                print "drill_point: %s, type: %s" % (drill_point, game_map.get_type(drill_point))
                print "can drill: %s" % game_map.can_drill_point(drill_point)
                if game_map.can_drill_point(drill_point) and self.curr_node.type != BlockType.PIPE:
                    print "Making a trap ...."
                    return '%s,%s' % ('ACT', nearest_ai_direction)

        if self.queue:
            print "take an action from queue: %s" % self.queue[0]
            move = self.queue.pop(0)
            if game_map.can_move_forward():
                print "move: %s" % move
                return move
            else:
                print "Can't move forward: %s" % get_opposite_direction(game_map.look_direction)
                return get_opposite_direction(game_map.look_direction)

        self.goldPos = game_map.find_nearest_type(game_map.me_point, BlockType.GOLD)

        if not self.goldPos:
            return self.get_no_gold_action(game_map)
        print "goldPos= %d %d " % (self.goldPos[0], self.goldPos[1])

        self.route = game_map.get_route(game_map.me_point, self.goldPos)
        print "path= %s" % self.route

        if self.route:
            next_point = self.route[0]
            revertible_points = game_map.get_revertible_points(next_point)
            revertible_points_types = [game_map.get_type(p) for p in revertible_points]
            print "next_point: %s: %s" % (next_point, game_map.matrix[next_point[0]][next_point[1]].type)
            print "revertible_point: %s" % revertible_points
            print "revertible_point node types: %s" % revertible_points_types
            if game_map.can_ai_get_me(next_point):
                print "AI can get me"
                return ''

            if self.should_drill(game_map):

                if self.is_gold_under_me(game_map):
                    print "Gold is under me"
                    turn = get_direction(self.me_point, next_point)
                    turn_back = get_opposite_direction(turn)
                    self.queue.append('%s,%s' % ('ACT', turn_back))
                    self.queue.append(turn_back)
                    return turn

                direction = get_direction(self.me_point, self.route[0])
                self.queue.append(direction)
                return '%s,%s' % ('ACT', direction)

            else:
                return get_direction(game_map.me_point, self.route.pop(0))

        else:
            print "WARNING - no route found"
            return ''

    def should_drill(self, game_map):
        if len(self.route) > 1:
            breakable_next_to_next = game_map.get_type(self.route[1]) == BlockType.BREAKABLE
            #breakable_next_to_next = game_map.matrix[self.route[1][0]][self.route[1][1]].type == BlockType.BREAKABLE
            x_point_of_block_is_next_to_me = abs(game_map.me_point[0] - self.route[1][0]) == 1
            ai_cant_get_me = BlockType.AI not in game_map.get_neighbor_nodes_types(self.me_point)
            return breakable_next_to_next and x_point_of_block_is_next_to_me and ai_cant_get_me

    def get_no_gold_action(self, game_map):
        print "WARNING - no gold found"
        if game_map.can_escape():
            print "Escape plan"
            self.queue.append(game_map.look_direction)
            return 'ACT'
        else:
            joints = game_map.matrix[game_map.me_point[0]][game_map.me_point[1]].joints
            if joints:
                index = random.randint(0, len(joints) - 1)
                return get_direction(game_map.me_point, joints[index].point)
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

    def get_surround_points(self, game_map):
        surround_points = []
        multipliers = [-1, 1]
        for i in range(1, 5):
            for multiplier in multipliers:
                x_position = self.me_point[0] + multiplier * i

                if x_position < game_map.size:
                    surround_points.append((x_position, self.me_point[1]))
        return surround_points


def is_player_chased(points, game_map):
    if BlockType.AI in [game_map.matrix[p[0]][p[1]].type for p in points]:
        return True
    return False


def get_nearest_ai_point(points, game_map):
    for point in points:
        if game_map.matrix[point[0]][point[1]].type == BlockType.AI:
            return point


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

    #print self.route[0], p
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


class Render():

    def __init__(self):
        print "Render is NOT used"

    def start(self):
        msg = None
        game_map = Map()
        solver = Solver()
        while True:
            time.sleep(0.1)
            if config.newMSG:
                msg = ws_client.AccessVarMSG(True, None)
                game_map.parse_msg(msg)
                game_map.create_graph_info()
                turn = solver.solve(game_map)
                print "turn: %s" % turn
                if turn in ['LEFT', 'RIGHT']:
                    if game_map.look_direction != turn:
                        game_map.look_direction = turn
                print "Look direction: %s" % game_map.look_direction
                ws_client.AccessVarResult(False, turn)

"""
TODO:
1. improve should_drill (alternative grapth where we can't move through breakable)
2. make sure player doesn't let ai catch him
3. better handling no gold found
4. improve logistics
"""