import numpy as np 


class Game :

    def __init__(self):
        print("Game class build ! ");
        self.board = np.zeros((3,3));

    def getState(self):
        return self.board;

    def doAction(self , moveNumber , value):
        self.board[int(moveNumber / 3)][int(moveNumber % 3)] = value;    
    
    def whoWins(self):
        agentRules =[
            [self.board[0][i] == 1 for i in range(3)],          #TRUE , TRUE , FALSE
            [self.board[1][i] == 1 for i in range(3)],          #FALSE , FALSE , FALSE
            [self.board[2][i] == 1 for i in range(3)],          #TRUE , TRUE , TRUE
            [self.board[i][0] == 1 for i in range(3)],
            [self.board[i][1] == 1 for i in range(3)],
            [self.board[i][2] == 1 for i in range(3)],
            [self.board[i][i] == 1 for i in range(3)],
            [self.board[i][2-i] == 1 for i in range(3)]        
        ]

        oponentRules =[
            [self.board[0][i] == 2 for i in range(3)],          #TRUE , TRUE , FALSE
            [self.board[1][i] == 2 for i in range(3)],          #FALSE , FALSE , FALSE
            [self.board[2][i] == 2 for i in range(3)],          #TRUE , TRUE , TRUE
            [self.board[i][0] == 2 for i in range(3)],
            [self.board[i][1] == 2 for i in range(3)],
            [self.board[i][2] == 2 for i in range(3)],
            [self.board[i][i] == 2 for i in range(3)],
            [self.board[i][2-i] == 2 for i in range(3)]        
        ]

        agentEvaluation = [all(ruleHandle) for ruleHandle in agentRules ]
        oponentEvaluation = [all(ruleHandle) for ruleHandle in oponentRules ]

        if(any(agentEvaluation)):
            return 10;
        elif(any(oponentEvaluation)):
            return -10;
        else:
            return 0;


    def isLocationMarked(self , loc):
        return True if self.board[int(loc/3)][int(loc%3)] == 0 else False



