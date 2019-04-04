import numpy as np 
import random 

def _isLocationFree(location):
    return True if(location == 0) else False


def predictMove(board):

    transposedBoard = np.transpose(board);
    movePredicted = -1;

    for i in range(3):
        temp = list(map(_isLocationFree , board[i]));        
        transTemp = list(map(_isLocationFree , transposedBoard[i]));        
        firstDiagonal = list(map(_isLocationFree , board.diagonal()));        
        secondDiagonal = list(map(_isLocationFree , np.diag(np.rot90(board))))            

        if(any(temp) and np.sum(board[i]) == 4):
            print("Ionel has just found an attack place ! {}".format(np.argmax(temp) + i*3))
            movePredicted = np.argmax(temp) + i*3;
        if(any(transTemp) and np.sum(transposedBoard[i]) == 4):
            print("Ionel has just found an attack place !")
            movePredicted = np.argmax(transTemp) + i*3;
        if(any(firstDiagonal) and np.sum(board.diagonal()) == 4):
            print("Ionel has just found an attack place !")
            movePredicted = np.argmax(firstDiagonal) + i*3;            
        if(any(secondDiagonal) and np.sum(np.diag(np.rot90(board))) == 4):
            print("Ionel has just found an attack place !")
            movePredicted = np.argmax(secondDiagonal) + i*3;

    if(movePredicted == -1):
        movePredicted = random.randint(0,9);
        while board[int(movePredicted / 3)][int(movePredicted % 3)] != 0:
            movePredicted = random.randint(0,9);
    

    return movePredicted;
