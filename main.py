import numpy as np 
import copy
import random
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from TicTacToe import Game
from IonelBoot import predictMove
import os 

learningRate = 0.0005
discountRate = 0.90
containerTransitions = list()

def rememberData(currentState , action , newState , reward):
    global containerTransitions

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

def loadDNN(modelPath):
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

    model.load_weights(modelPath)
    return model;

#currentState shape is (1,9)
#action is index of max prob 
#reward = [-100 , -10 , 0 , +10]
def train_short_term_memory(currentState , action , newState , reward , finished , networkModel , wasRandomMove):
    global discountRate
    global learningRate

    nrOfEpochs = 1 if reward > -100 else 100 ;

    if((wasRandomMove == True) and reward < -100):
        nrOfEpochs = 10;

    if(finished == False):

        probabilitiesInCurrentState = networkModel.predict(currentState)[0];       #[[0.1 , 0.11 , 0.2 , ... , 0.05]]
        probOfTookenAction = probabilitiesInCurrentState[action]                #0.2

        probOfWiningInNextState = networkModel.predict(newState)[0];       #[[0.14 , 0.5 , .... , 0.15]]
        maxProbOfWinningInNextState = np.max(probOfWiningInNextState);  #0.5

        newProbForTookenAction = probOfTookenAction + learningRate*(reward + discountRate*maxProbOfWinningInNextState);

        target = copy.deepcopy(probabilitiesInCurrentState);
        target[action] = newProbForTookenAction;

        networkModel.fit(probabilitiesInCurrentState.reshape(1,9) , target.reshape(1,9) , epochs = nrOfEpochs , verbose = 0);

    else:
        probabilitiesInCurrentState = networkModel.predict(currentState)[0];       #[[0.1 , 0.11 , 0.2 , ... , 0.05]]
        probOfTookenAction = probabilitiesInCurrentState[action]                    #0.2
        newProbForTookenAction = probOfTookenAction + learningRate*reward;

        target = copy.deepcopy(probabilitiesInCurrentState);
        target[action] = newProbForTookenAction;
        networkModel.fit(probabilitiesInCurrentState.reshape(1,9) , target.reshape(1,9) , epochs = nrOfEpochs , verbose = 0);

def train_network_on_stored_data(networkModel):
    global containerTransitions

    if(len(containerTransitions) > 200):
        batch = random.sample(containerTransitions , 200);
    else:
        batch = containerTransitions;

    for oldState , act , state , r in batch:
        train_short_term_memory(oldState , act , state , r , True if r !=0 else False , networkModel , False);




def main():

    nrOfGames = 5000
    luckyPercent = 70;
    
    game = Game();
    #game.doAction(3,2);    
    #game.doAction(4,2);  
    print(game.whoWins());
    fileName = "model.h5"

    if(os.path.isfile(fileName)):
        dnnModel = loadDNN(fileName);
    else:
        dnnModel = buildDNN();
        for i in range(nrOfGames):
            counter = 0;
            while(True):
                counter = counter + 1;
                randomMove = False;

                print("Game number: {}".format(i));
                #Get current state of the game 
                currentState = np.reshape(game.getState() , (1,9));
                backupCurrentState = copy.deepcopy(currentState)

                if(random.randint(1,100) < int(luckyPercent - (luckyPercent / (i+1)))):
                    print("Lucky branch !");
                    move = random.randint(0,8);
                    randomMove = True;
                else:
                    print("Knowledge branch !");
                    #Predict next move based on current state             
                    output = dnnModel.predict(currentState);        #[0.12 , 0.14 ... , 0.22]

                    #Take the index of the maximum probability
                    move = np.argmax(output);                       #8

                if(game.isLocationMarked(move)):
                    print("Location overwritten !")
                    #Remember transition 
                    rememberData(backupCurrentState , move , backupCurrentState , -1000); 
                    train_short_term_memory(backupCurrentState,move,backupCurrentState,-1000 , True , dnnModel , randomMove);
                    break;                                

                #Agent is doing the move on the board !
                game.doAction(move , 1);        
                #Check if agent won 
                r = game.whoWins();
                if(r != 0):
                    tempNewState = game.getState();
                    rememberData(backupCurrentState , move , np.reshape(tempNewState , (1,9)) , r);
                    train_short_term_memory(backupCurrentState , move , np.reshape(tempNewState , (1,9)) , r , True , dnnModel , randomMove);
                    break;

                if(game.areThereAnyFreeSpots()):
                    #Ionel is doing the move because agent hasn't won yet 
                    game.doAction(predictMove(game.getState()) , 2);
                    newState = np.reshape(game.getState() , (1,9));
                    r = game.whoWins();
                    if(r != 0):
                        rememberData(backupCurrentState , move , newState , r)
                        train_short_term_memory(backupCurrentState , move , newState , r , True , dnnModel , randomMove);
                        break;
                else:
                    break;


                train_short_term_memory(backupCurrentState , move , newState , 0 , False , dnnModel , randomMove);
                rememberData(backupCurrentState , move , newState , 0);

                if(counter > 50):
                    counter = 0;

            game.resetGame();       
            train_network_on_stored_data(dnnModel);

    
        dnnModel.save_weights(fileName)
    # TBD : Train_network_on_stored_data

    testStates = [
        [1 , 0 , 2 , 0 , 1 , 0 , 2 , 0 , 0],
        [1 , 0 , 2 , 0 , 0 , 0 , 2 , 0 , 1],
        [1 , 1 , 0 , 2 , 2 , 0 , 0 , 0 , 0],
        [1 , 0 , 1 , 2 , 2 , 0 , 0 , 0 , 0],
        [1 , 0 , 2 , 1 , 2 , 0 , 0 , 0 , 0]
        ]

    #testStates = np.array(testStates);

    expectedMove = [8,4,2,1,6]

    #interm = np.array([1,0,2,0,1,0,2,0,0]);
    #interm = interm.reshape((1,9))  
    #print(dnnModel.predict(interm)[0]) 
    #a = np.argmax(dnnModel.predict(interm)[0]);
    #print(a)
    
    #train_short_term_memory(np.array([1,0,2,0,1,0,2,0,0]).reshape((1,9)) , 6 , np.array([1,0,2,0,1,0,2,0,0]).reshape((1,9)) , -1000000000 , True , dnnModel);
    #print(dnnModel.predict(interm)[0]);
    #a = np.argmax(dnnModel.predict(interm)[0]);
    #print(a)

    for i in range(len(testStates)):       
        print(dnnModel.predict(np.array(testStates[i]).reshape((1,9)))); 
        a = np.argmax(dnnModel.predict(np.array(testStates[i]).reshape((1,9)))[0]);
        print(a);
        if(a == expectedMove[i]):
            print("Success !");
        else:
            print("Fail !");



if __name__ == "__main__":
    main()


    