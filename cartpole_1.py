import numpy as np
import time

def to_cut(readgm, epi = 2):
    all_cut = []
    for i in readgm(epi = epi):
        all_cut.append(i[0])
    return all_cut

def readgmc(gamename = "CartPole-v1", epi = 2, maxsteps = 1000, mapaction = []):
    """
    Python function for importing the gym game data set.  It returns an iterator
    of 2-tuples with the first element being the 2D array of pixel data for the given 
    screen and the second element
    being a reward(and actions)
    """
    import gym
    
    env = gym.make(gamename)
    
    for i_episode in range(epi):
        observation_p = env.reset()
        for t in range(maxsteps):
            env.render
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            yield observation_p, reward, action
            observation_p = observation
            # print(action, reward)
            if done:
                #yield 0,0,t+1
                #print("finished after {} steps".format(t+1), info)
                break
                
def readgmc1(gamename = "CartPole-v1", epi = 2, maxsteps = 1000, mapaction = []):
    """
    Python function for importing the gym game data set.  It returns an iterator
    of 2-tuples with the first element being the 2D array of pixel data for the given 
    screen and the second element
    being a reward(and actions)
    """
    import gym
    
    env = gym.make(gamename)
    
    for i_episode in range(epi):
        observation_p = env.reset()
        for t in range(maxsteps):
            env.render
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            yield observation_p, reward, action
            observation_p = observation
            # print(action, reward)
            if done:
                yield 0,0,t+1
                #to tell map_long the lenth of an epi, and fit the format of output
                #print("finished after {} steps".format(t+1), info)
                break
                
def map_long(readgm, matrix, findbmu, thres, epi = 2):
    all_long = [[0,0,0],[0,0,0]]
    temp_one = []
    for i in readgm(epi = epi):
        if i[1] != 0:
            bmu = findbmu(i[0],matrix)
            temp_one.append([bmu,0,int((i[2]-1/2)*2)])
            #change actions to 1 and -1
        else:
            temp_one = np.array(temp_one)
            temp_one[:,1] = i[2]
            temp_one = temp_one[:-(thres+1)]
            #del some last bad experience 
            all_long = np.concatenate((all_long, temp_one))
            temp_one = []
    all_long = all_long[2:]  
    print('agent array is',all_long)
    return all_long

def findbmu(ddata,matrix):
    d = np.dot(matrix, ddata.T)
    bmu = np.argmax(d, axis = 0)
    return bmu

def karma_ca(po, mapsize):
    map_karma = np.zeros(mapsize)
    for i in range(mapsize):
        kmat = po[np.where(po[:,0] == i)]
        map_karma[i] = np.dot(kmat[:,1],kmat[:,2])
        
    karma1 = []
    for i in map_karma:
        if i > 0:
            karma1.append(1)
        else:
            karma1.append(0)
    return karma1
        
def testgm(matrix, karma, gamename = "CartPole-v1", epi = 1, maxsteps = 3000):
    """
    Python function for running the gym game data set, with prepared karma array. 
    """
    import gym
    
    env = gym.make(gamename)
    
    for i in range(epi):
        observation = env.reset()
        for t in range(maxsteps):
            env.render()
            #bmu = 1
            bmu = findbmu(observation, matrix)
            action = np.sign(karma[int(bmu)])
            time.sleep(.005)
            #for watchers
            #action = 0
            observation, reward, done, info = env.step(action)
            if done:
                print("finished after {} steps".format(t+1))
                break
    env.close()
    


