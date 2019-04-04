import numpy as np 
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from TicTacToe import Game
from IonelBoot import predictMove

def buildDNN():
    model = Sequential()
    model.add(Dense(input_dim=9, activation="relu", units=120))
    model.add(Dropout(0.15));
    model.add(Dense(activation="relu", units=120))
    model.add(Dropout(0.15));
    model.add(Dense(activation="relu", units=120))
    model.add(Dropout(0.15));
    model.add(Dense(activation="softmax", units = 9))
    optimizer = Adam(lr=0.001);
    model.compile(optimizer , loss="mse");

    return model;


def main():

    dnnModel = buildDNN();
    game = Game();
    game.doAction(3,2);    
    game.doAction(4,2);  
    print(game.whoWins());

    while(game.whoWins() == 0):

        currentState = np.reshape(game.getState() , (1,9));
        output = dnnModel.predict(currentState);
        move = np.argmax(output);
        if(game.isLocationMarked(move)):
            tempOutput = output;
            del tempOutput[move];
            

        game.doAction(move , 1);        
        game.doAction(predictMove(game.getState()) , 2);


        print(output);
        print(move);
        break;



if __name__ == "__main__":
    main()


    