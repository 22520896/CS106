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
    beginBox = PosOfBoxes(gameState) # Lấy vị trí khởi đầu của các thùng e.g. ((3,1), (3,4))
    beginPlayer = PosOfPlayer(gameState) # Lấy vị trí khởi đầu của người chơi e.g. (4,1)
   
    startState = (beginPlayer, beginBox) # Tạo trạng thái khởi đầu gồm vị trí ban đầu của người chơi và các thùng e.g. ((4,1) , ((3,1), (3,4))
    frontier = collections.deque([[startState]]) # Khởi tạo hàng đợi frontier sẽ chứa các NODE với NODE đầu tiên chỉ gồm startState
    exploredSet = set() # Khởi tạo tập hợp chứa các trạng thái đã được xét
    actions = collections.deque([[0]]) # Khởi tạo hàng đợi mà MỖI PHẦN TỬ LÀ MỘT ĐƯỜNG ĐI (dãy các hành động) ứng với một NODE trong frontier theo thứ tự
                                       # e.g. ([0, ‘l’, ‘r’], [0, ‘R’, l])
    temp = [] # List sẽ chứa đường đi cần tìm từ startState đến goalState (trạng thái kết thúc)
    ### CODING FROM HERE ###
    while frontier: # Lặp qua hàng đợi frontier để mở rộng các node
        node = frontier.popleft() # Lấy ra khỏi frontier node trái cùng (phần tử vào hàng đợi sớm nhất) 
        # Ta sẽ xét trạng thái cuối cùng trong node này: node[-1] - gọi trạng thái này là TRẠNG THÁI HIỆN TẠI 
        node_action = actions.popleft() # Lấy ra khỏi actions phần tử trái cùng (phần tử vào hàng đợi sớm nhất)
                                        # là đường đi tương ứng với node ở trên, e.g. [0, ‘l’, ‘r’]
        if isEndState(node[-1][-1]): # Kiểm tra xem TRẠNG THÁI HIỆN TẠI đó có phải là goalState hay không 
            temp += node_action[1:] # Nếu đúng, ta thêm đường đi tương ứng với node đang xét vào temp và dừng vòng lặp
            break
        if node[-1] not in exploredSet: # Ngược lại, kiểm tra TRẠNG THÁI HIỆN TẠI đã được xét chưa
            exploredSet.add(node[-1]) # Nếu chưa được xét, thêm TRẠNG THÁI HIỆN TẠI vào tập exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động (action) hợp lệ từ TRẠNG THÁI HIỆN TẠI 
                                                                  # action có dạng (0, -1, 'l')
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
    # Cập nhật vị trí mới của người chơi và các thùng dựa trên TRẠNG THÁI HIỆN TẠI và action được TRẠNG THÁI MỚI (gồm vị trí mới của người chơi và thùng)
                
                if isFailed(newPosBox): # Kiểm tra TRẠNG THÁI MỚI có dẫn tới thất bại không
                    continue # Nếu đúng, bỏ qua vòng lặp hiện tại và tới vòng lặp tiếp theo
                frontier.append(node + [(newPosPlayer, newPosBox)]) 
                # Ngược lại, thêm TRẠNG THÁI MỚI vào cuối node đang xét để được node mới rồi thêm node mới này vào frontier 
                actions.append(node_action + [action[-1]]) 
    # Thêm action[-1] (e.g 'l') vào cuối đường đi node_action để được đường đi từ startState đến TRẠNG THÁI MỚI rồi thêm đường đi này vào actions 
    return temp # Trả về đường đi từ startState đến goalState e.g. [‘l’, ‘r’, 'R','u','U'], hoặc trả về mảng rỗng nếu không thể tìm thấy đường đi nào

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
    while not frontier.isEmpty(): # Lặp qua hàng đợi ưu tiênfrontier để mở rộng các node
        node = frontier.pop() # Lấy ra khỏi frontier node có độ ưu tiên cao nhất (chi phí thấp nhất)
        # Ta sẽ xét trạng thái cuối cùng trong node này: node[-1] - gọi trạng thái này là TRẠNG THÁI HIỆN TẠI
        node_action = actions.pop() # Lấy ra khỏi actions đường đi có độ ưu tiên cao nhất (chi phí thấp nhất) ứng với node ở trên
        if isEndState(node[-1][-1]): # Kiểm tra xem TRẠNG THÁI HIỆN TẠI đó có phải là goalState hay không 
            temp += node_action[1:] # Nếu đúng, ta thêm đường đi tương ứng với node đang xét vào temp và dừng vòng lặp
            break
        if node[-1] not in exploredSet: # Ngược lại, kiểm tra TRẠNG THÁI HIỆN TẠI đã được xét chưa.
            exploredSet.add(node[-1]) # Nếu chưa được xét, thêm TRẠNG THÁI HIỆN TẠI vào tập exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Duyệt qua tất cả các hành động (action) hợp lệ từ TRẠNG THÁI HIỆN TẠI 
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
            # Cập nhật vị trí mới của người chơi và các hộp dựa trên TRẠNG THÁI HIỆN TẠI và action ta được TRẠNG THÁI MỚI
                if isFailed(newPosBox): # Kiểm tra TRẠNG THÁI MỚI có dẫn tới thất bại không
                    continue # Nếu đúng, bỏ qua vòng lặp hiện tại và tới vòng lặp tiếp theo
                # Ngược lại
                new_action = node_action + [action[-1]] 
                # Thêm action[-1] (e.g 'l') vào cuối đường đi node_action để được ĐƯỜNG ĐI MỚI từ startState đến TRẠNG THÁI MỚI
                new_cost = cost(new_action[1:]) # Tính chi phí của ĐƯỜNG ĐI MỚI ở trên 
                frontier.push(node + [(newPosPlayer, newPosBox)], new_cost) 
                # thêm TRẠNG THÁI MỚI vào cuối node đang xét để được node mới rồi thêm node mới này vào frontier, độ ưu tiên là chi phí ĐƯỜNG ĐI MỚI
                actions.push(new_action, new_cost) #Thêm ĐƯỜNG ĐI MỚI vào actions với độ ưu tiên là chi phí của nó 
    return temp # Trả về đường đi có chi phí thấp nhất từ startState đến goalState, hoặc trả về mảng rỗng nếu không thể tìm thấy đường đi nào

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
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
