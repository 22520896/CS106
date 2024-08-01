import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = collections.deque([[0]])
    temp = []
    ### CODING FROM HERE ###

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # Lấy vị trí khởi đầu của các thùng e.g. ((3,1), (3,4))
    beginPlayer = PosOfPlayer(gameState) # Lấy vị trí khởi đầu của người chơi e.g. (4,1)
    startState = (beginPlayer, beginBox) # Tạo trạng thái khởi đầu gồm vị trí ban đầu của người chơi và các thùng
    frontier = PriorityQueue() # Khởi tạo hàng đợi ưu tiên sẽ chứa các NODE 
                               # với độ ưu tiên là chi phí đường đi từ stateStart đến trạng thái cuối cùng trong NODE
    frontier.push([startState], 0) # Thêm node đâu tiên chỉ gồm startState vào frontier với chi phí là 0
    exploredSet = set() # Khởi tạo tập hợp chứa các trạng thái đã được xét
    actions = PriorityQueue() # Khởi tạo hàng đợi ưu tiên mà MỖI PHẦN TỬ LÀ MỘT ĐƯỜNG ĐI ứng với một node trong frontier theo thứ tự
                              # với độ ưu tiên là chi phí đường đi đó
    actions.push([0], 0)  # Thêm vào actions đường đi đầu tiên (chưa có hành động nào) với chi phí là 0
    temp = [] # List sẽ chứa đường đi cần tìm từ startState đến goalState
    ### CODING FROM HERE ###
    while len(frontier.Heap) > 0: # Lặp qua frontier để mở rộng các node
        node = frontier.pop() # Lấy ra khỏi frontier node có độ ưu tiên cao nhất (chi phí thấp nhất)
        # Ta sẽ xét trạng thái cuối cùng trong node này: node[-1] - gọi trạng thái này là TRẠNG THÁI HIỆN TẠI
        node_action = actions.pop() # Lấy ra khỏi actions đường đi có độ ưu tiên cao nhất (chi phí thấp nhất) ứng với node ở trên
        if isEndState(node[-1][-1]): # Kiểm tra xem TRẠNG THÁI HIỆN TẠI đó có phải là goalState hay không 
            temp += node_action[1:] # Nếu đúng, ta thêm đường đi tương ứng với node đang xét vào temp và dừng vòng lặp
            break
        if node[-1] not in exploredSet: # Ngược lại, kiểm tra TRẠNG THÁI HIỆN TẠI đã được xét chưa
            exploredSet.add(node[-1]) # Nếu chưa được xét, thêm TRẠNG THÁI HIỆN TẠI vào tập exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động (action) hợp lệ từ TRẠNG THÁI HIỆN TẠI 
            
            # Cập nhật vị trí mới của người chơi và các thùng dựa trên TRẠNG THÁI HIỆN TẠI và action ta được TRẠNG THÁI MỚI
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
                if isFailed(newPosBox): # Kiểm tra TRẠNG THÁI MỚI có dẫn tới thất bại không
                    continue # Nếu đúng, bỏ qua vòng lặp hiện tại và tới vòng lặp tiếp theo
                # Ngược lại
                # Thêm action[-1] (e.g 'l') vào cuối đường đi node_action để được ĐƯỜNG ĐI MỚI từ startState đến TRẠNG THÁI MỚI
                new_action = node_action + [action[-1]] 
                new_cost = cost(new_action[1:]) # Tính chi phí của ĐƯỜNG ĐI MỚI ở trên 

                # Thêm TRẠNG THÁI MỚI vào cuối node đang xét để được node mới rồi thêm node mới này vào frontier, độ ưu tiên là chi phí ĐƯỜNG ĐI MỚI
                frontier.push(node + [(newPosPlayer, newPosBox)], new_cost) 
                actions.push(new_action, new_cost) # Thêm ĐƯỜNG ĐI MỚI vào actions với độ ưu tiên là chi phí của nó 

    return temp # Trả về đường đi có chi phí thấp nhất từ startState đến goalState, hoặc trả về mảng rỗng nếu không thể tìm thấy đường đi nào

def heuristic(posPlayer, posBox): # Manhattan
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance

'''def heuristic(posPlayer, posBox): # Euclid
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += ((sortposBox[i][0] - sortposGoals[i][0])**2 + (sortposBox[i][1] - sortposGoals[i][1])**2)**0.5
    return distance'''

def aStarSearch(gameState):
    """Implement aStarSearch approach"""
    # start =  time.time()
    beginBox = PosOfBoxes(gameState) # Lấy vị trí khởi đầu của các thùng e.g. ((3,1), (3,4))
    beginPlayer = PosOfPlayer(gameState) # Lấy vị trí khởi đầu của người chơi e.g. (4,1)
    temp = [] # List sẽ chứa đường đi cần tìm từ start_state đến goalState
    start_state = (beginPlayer, beginBox)  # Tạo trạng thái khởi đầu gồm vị trí khởi đầu của người chơi và các thùng

    # Gọi f(n) là tổng chi phí đường đi từ start_state đến trạng thái cuối cùng trong node n
                            # và chi phí ước lượng (bằng hàm heuristic) từ trạng thái cuối cùng này đến goalState

    frontier = PriorityQueue() # Khởi tạo hàng đợi ưu tiên sẽ chứa các NODE với độ ưu tiên là f của các node đó
    # Thêm node đầu tiên chỉ gồm start_state vào frontier với độ ưu tiên = heurictis(start_state) + 0
    frontier.push([start_state], heuristic(beginPlayer, beginBox)) 
    exploredSet = set() # Khởi tạo tập hợp chứa các trạng thái đã được xét
    actions = PriorityQueue() # Khởi tạo hàng đợi ưu tiên mà MỖI PHẦN TỬ LÀ MỘT ĐƯỜNG ĐI ứng với một node trong frontier theo thứ tự
                              # Độ ưu tiên 1 đường đi của actions cũng chính là độ ưu tiên của node tương ứng với nó trong frontier  
    actions.push([0], heuristic(beginPlayer, start_state[1])) # Thêm vào actions đường đi đầu tiên (chưa có hành động nào) ứng với node đầu tiên frontier
    while len(frontier.Heap) > 0: # Lặp qua frontier để mở rộng các node
        node = frontier.pop() # Lấy ra khỏi frontier node có độ ưu tiên cao nhất (f thấp nhất)
        # Ta sẽ xét trạng thái cuối cùng trong node này: node[-1] - gọi trạng thái này là TRẠNG THÁI HIỆN TẠI
        node_action = actions.pop() # Lấy ra khỏi actions đường đi có độ ưu tiên cao nhất ứng với node ở trên
        if isEndState(node[-1][-1]): # Kiểm tra xem TRẠNG THÁI HIỆN TẠI đó có phải là goalState hay không 
            temp += node_action[1:] # Nếu đúng, ta thêm đường đi tương ứng với node đang xét vào temp và dừng vòng lặp
            break

        ### CONTINUE YOUR CODE FROM HERE
        if node[-1] not in exploredSet: # Ngược lại, kiểm tra TRẠNG THÁI HIỆN TẠI đã được xét chưa.
            exploredSet.add(node[-1]) # Nếu chưa được xét, thêm TRẠNG THÁI HIỆN TẠI vào tập exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động (action) hợp lệ từ TRẠNG THÁI HIỆN TẠI 

                # Cập nhật vị trí mới của người chơi và các thùng dựa trên TRẠNG THÁI HIỆN TẠI và action ta được TRẠNG THÁI MỚI
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
                if isFailed(newPosBox): # Kiểm tra TRẠNG THÁI MỚI có dẫn tới thất bại không
                    continue # Nếu đúng, bỏ qua vòng lặp hiện tại và tới vòng lặp tiếp theo
                # Ngược lại
                # Thêm action[-1] (e.g 'l') vào cuối đường đi node_action để được ĐƯỜNG ĐI MỚI từ start_state đến TRẠNG THÁI MỚI
                new_action = node_action + [action[-1]] 
                new_f =  heuristic(newPosPlayer, newPosBox) + cost(new_action[1:]) # Tính f của NODE MỚI chứa TRẠNG THÁI MỚI trên
                # Thêm TRẠNG THÁI MỚI vào cuối node đang xét để được NODE MỚI rồi thêm NODE MỚI này vào frontier, độ ưu tiên là new_f
                frontier.push(node + [(newPosPlayer, newPosBox)], new_f) 
                actions.push(new_action, new_f) # Thêm ĐƯỜNG ĐI MỚI vào actions với độ ưu tiên là new_f  

    # end =  time.time()
                
    return temp # Trả về đường đi từ start_state đến goalState, hoặc trả về mảng rỗng nếu không thể tìm thấy đường đi nào

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'astar':
        result = aStarSearch(gameState)        
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
