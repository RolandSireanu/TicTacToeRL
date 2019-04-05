import numpy as np 
import copy
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from TicTacToe import Game
from IonelBoot import predictMove

learningRate = 0.0005
discountRate = 0.90
containerTransitions = list()

def rememberData(currentState , action , newState , reward):
    containerTransitions.append([currentState , action , newState , reward]);


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

#currentState shape is (1,9)
#action is index of max prob 
#reward = [-100 , -10 , 0 , +10]
def train_short_term_memory(currentState , action , newState , reward , finished , networkModel):
    global discountRate
    global learningRate

    probabilitiesInCurrentState = networkModel.predict(currentState)[0];       #[[0.1 , 0.11 , 0.2 , ... , 0.05]]
    probOfTookenAction = probabilitiesInCurrentState[action]                #0.2

    probOfWiningInNextState = networkModel.predict(newState)[0];       #[[0.14 , 0.5 , .... , 0.15]]
    maxProbOfWinningInNextState = np.max(probOfWiningInNextState);  #0.5

    newProbForTookenAction = probOfTookenAction + learningRate*(reward + discountRate*maxProbOfWinningInNextState);

    target = probabilitiesInCurrentState;
    target[action] = newProbForTookenAction;

    networkModel.fit(probabilitiesInCurrentState.reshape(1,9) , target.reshape(1,9) , epochs = 1 , verbose = 0);



def main():

    dnnModel = buildDNN();
    game = Game();
    #game.doAction(3,2);    
    #game.doAction(4,2);  
    print(game.whoWins());

    while(game.whoWins() == 0):

        #Get current state of the game 
        currentState = np.reshape(game.getState() , (1,9));
        backupCurrentState = copy.deepcopy(currentState)

        #Predict next move based on current state 
        output = dnnModel.predict(currentState);        #[0.12 , 0.14 ... , 0.22]

        #Take the index of the maximum probability
        move = np.argmax(output);                       #8
        if(game.isLocationMarked(move)):
            #Remember transition 
            rememberData(backupCurrentState , move , backupCurrentState , -100);            
            break;                    

        #Agent is doing the move on the board !
        game.doAction(move , 1);        
        #Check if agent won 
        r = game.whoWins();
        if(r != 0):
            tempNewState = game.getState();
            rememberData(backupCurrentState , move , np.reshape(tempNewState , (1,9)) , r);
            train_short_term_memory(backupCurrentState , move , tempNewState , r , True , dnnModel);
            break;

        #Ionel is doing the move because agent hasn't won yet 
        game.doAction(predictMove(game.getState()) , 2);

        newState = np.reshape(game.getState() , (1,9));

        train_short_term_memory(backupCurrentState , move , newState , 0 , False , dnnModel);


        print(output);
        print(move);
        break;

    # TBD : Train_network_on_stored_data

if __name__ == "__main__":
    main()


    