if __name__== "__main__":
    
    import gymnasium as gym
    import time
    
    import os
    import copy

    from timeit import default_timer as timer

    import numpy as np
    import pickle

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout 
    from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D, LeakyReLU, UpSampling2D, Conv2D, MaxPooling2D 
    from tensorflow.keras.layers import RandomFlip, RandomTranslation, RandomRotation, RandomZoom
    from tensorflow.keras.models import Sequential, Model 
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.models import save_model, load_model
    from tensorflow.keras import initializers

    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from tensorflow.keras import backend as K

    import random

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    

    def save_optimizer_state(optimizerIn, save_path, save_name):
        '''
        Save keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object.
        save_path --- Path to save location.
        save_name --- Name of the .npy file to be created.

        '''

        # Create folder if it does not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save weights
        np.save(os.path.join(save_path, save_name), optimizerIn.get_weights())

        return

    def load_optimizer_state(optimizer, load_path, load_name, model_train_vars):
        '''
        Loads keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object to be loaded.
        load_path --- Path to save location.
        load_name --- Name of the .npy file to be read.
        model_train_vars --- List of model variables (obtained using Model.trainable_variables)

        '''

        # Load optimizer weights
        opt_weights = np.load(os.path.join(load_path, load_name)+'.npy', allow_pickle=True)

        # dummy zero gradients
        zero_grads = [tf.zeros_like(w) for w in model_train_vars]
        # save current state of variables
        saved_vars = [tf.identity(w) for w in model_train_vars]

        # Apply gradients which don't do nothing with Adam
        optimizer.apply_gradients(zip(zero_grads, model_train_vars))

        # Reload variables
        [x.assign(y) for x,y in zip(model_train_vars, saved_vars)]

        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)


        return

    def critic_loss(discounted_rewards, value_est, critic_loss_weight_in):
        return tf.cast(tf.reduce_mean(keras.losses.mean_squared_error(discounted_rewards, value_est)) * critic_loss_weight_in, tf.float64)

    def entropy_loss(policy_logits, ent_discount_val):
        #probs = tf.nn.softmax(policy_logits)
        probs = policy_logits
        entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(probs, probs))
        return entropy_loss * ent_discount_val


    def actor_loss(advantages, old_probs, action_inds, policy_logits, clipValIn):
        
        probs = policy_logits
        new_probs = tf.gather_nd(probs, action_inds)
        ratio = new_probs / old_probs

        policy_loss = -tf.reduce_mean(tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - clipValIn, 1.0 + clipValIn) * advantages,
        ))
        
        return policy_loss

    def train_model(modelIn, action_inds, old_probs, states, advantages, discounted_rewards, optimizer, critic_loss_weight_in, ent_discount_val, clipValIn):
        with tf.GradientTape() as tape:
            values, policy_logits = policyModel1(states)

            policy_logits = tf.cast(policy_logits, dtype=tf.float64)

            act_loss = actor_loss(advantages, old_probs, action_inds, policy_logits, clipValIn)
            ent_loss = entropy_loss(policy_logits, ent_discount_val)
            c_loss = critic_loss(discounted_rewards, values, critic_loss_weight_in)
             
            tot_loss = act_loss + ent_loss + c_loss
        grads = tape.gradient(tot_loss, modelIn.trainable_variables)
        optimizer.apply_gradients(zip(grads, modelIn.trainable_variables))
        return tot_loss, c_loss, act_loss, ent_loss

    def get_advantages(rewards, dones, values, gammaIn):
        discounted_rewards = np.array(rewards)

        for t in reversed(range(0, len(rewards) - 1, 1)):
            discounted_rewards[t] = rewards[t] + (gammaIn * discounted_rewards[t+1] * (1-dones[t]))
        #discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - np.stack(values)[:, 0]
        # standardise advantages
        advantages -= np.mean(advantages)
        advantages /= (np.std(advantages) + 1e-10)
        
        # standardise rewards too
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        return discounted_rewards, advantages





    def createPolicyModel(imageShapeIn, actionNumIn):

        modelPolInput = Input(shape=imageShapeIn)

        modelPol = Conv2D(32, (3, 3), strides=2, padding = "same", activation="tanh")(modelPolInput) 
        modelPol = Conv2D(32, (3, 3), strides=2, padding = "same", activation="tanh")(modelPol) 
        modelPol = Conv2D(32, (3, 3), strides=2, padding = "same", activation="tanh")(modelPol)
        modelPol = Conv2D(32, (3, 3), strides=2, padding = "same", activation="tanh")(modelPol) 

        modelPol = Flatten()(modelPol)
        #modelPolDense1 = Dense(128, activation = 'relu')(modelPol)
        #modelPolDenseVal1 = Dense(32, activation = 'relu')(modelPolDense1)
        #modelPolDenseAct1 = Dense(32, activation = 'relu')(modelPolDense1)

        modelPolDense1 = Dense(256, activation = 'tanh')(modelPol)
        modelPolDenseVal1 = Dense(256, activation = 'tanh')(modelPolDense1)
        modelPolDenseAct1 = Dense(256, activation = 'tanh')(modelPolDense1)

        modelVal = Dense(1, activation = 'linear')(modelPolDenseVal1)
        modelAct = Dense(actionNumIn, activation = 'softmax')(modelPolDenseAct1)

        modelPolRet = Model(modelPolInput, [modelVal, modelAct])

        return modelPolRet


    numHyperParamTests = 1
    numAveragingRuns = 1

    learningRateQList = [0.0001, 0.00075, 0.001, 0.0005, 0.0005, 0.001, 0.005, 0.005, 0.005, 0.005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.008, 0.01, 0.01, 0.01, 0.01]
    learningRateQDecList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    rewardDiscountFactorList = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    rewardCumulateFactorList = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]

    entropyFactorList = [0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    evalFactorList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ppoFactorList = [0, 0, 0, 0, 0, 0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0]

    evalLearnRateList = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    evalDiscountFactorList = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]

    batchGradAcumNumList = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    trainOrTestFlag = 1

    numEpisodes = 1

    loadPolicyFlag = 0
    savePolicyFlag = 1

    if(trainOrTestFlag == 1):

        numEpisodes = 1

        loadPolicyFlag = 0
        savePolicyFlag = 1

    else:

        numEpisodes = 1

        loadPolicyFlag = 1
        savePolicyFlag = 0

    

    environmentsNum = 4

    if(trainOrTestFlag == 1):

        environmentsNum = 4

    else:

        environmentsNum = 1 

    channelNum = 3
    imageStackNum = 4
    imageInputShape = (210, 160, channelNum)

    policyImageShape = (210, 160, channelNum * imageStackNum)
    policyActionNum = 4

    #parallelDatastoreNum = 10
    parallelDatastoreNum = environmentsNum

    numTrainEpochs = 5
    ent_discount_rate = 0.995
    ent_discount_val = 0.01
    ppoClipVal = 0.2
    criticLossWeight = 0.5

    stepLimit = 1000
    stepLimitOffset = 0

 
    flattenedBatchNum = (environmentsNum * stepLimit)

    ppoBatchSize = 50
    
    outputShape = 16

    policySaveDirectory = "./policySaveDirectory/"

    policyModel1 = createPolicyModel(policyImageShape, policyActionNum)
    policyModel1.compile()



    IL = 128 #input layer nodes
    
    OL = policyActionNum #output layer nodes


    EL = 1 #value output nodes



    inValImageBatch = np.zeros((environmentsNum, imageInputShape[0], imageInputShape[1], imageInputShape[2]), dtype = float)
    inValImageOld1Batch = np.zeros((environmentsNum, imageInputShape[0], imageInputShape[1], imageInputShape[2]), dtype = float)
    inValImageOld2Batch = np.zeros((environmentsNum, imageInputShape[0], imageInputShape[1], imageInputShape[2]), dtype = float)
    inValImageOld3Batch = np.zeros((environmentsNum, imageInputShape[0], imageInputShape[1], imageInputShape[2]), dtype = float)

    inValAllImageBatch = np.zeros((environmentsNum, policyImageShape[0], policyImageShape[1], policyImageShape[2]), dtype = float)

    outValBatch = np.zeros((environmentsNum, OL), dtype = float)

    evalValBatch = np.zeros((environmentsNum, EL), dtype = float)

    rewardsBatch = np.zeros((environmentsNum, 1), dtype = float)
    donesBatch = np.zeros((environmentsNum, 1), dtype = float)


    bufAdded = 0
    bufCounter = 0
    bufFinished = 0

    envsList = []
    resetsList = []
    actionsList = []
    observationsList = []

    inValAllImageBuf = np.zeros((1, policyImageShape[0], policyImageShape[1], policyImageShape[2]), dtype = float)

    outValBuf = np.zeros((1, OL), dtype = float)
    evalValBuf = np.zeros((1, EL), dtype = float)

    advanValBuf = np.array(EL, dtype = float)

    actionBuf = np.zeros(1, dtype = float)
    rewardBuf = np.zeros(1, dtype = float)
    doneBuf = np.zeros(1, dtype = float)
    policyStartFinish = np.zeros(1, dtype = float)

    nextAction = 0



    ###############################################


    if(trainOrTestFlag == 1):

        #envs = gym.vector.AsyncVectorEnv([lambda: gym.make('SpaceInvaders-v4', render_mode="rgb_array")] * environmentsNum, shared_memory=False)
        envs = gym.vector.AsyncVectorEnv([lambda: gym.make('SpaceInvaders-v0', render_mode="rgb_array")] * environmentsNum, shared_memory=False)

    else:

        #envs = gym.vector.AsyncVectorEnv([lambda: gym.make('SpaceInvaders-v4', render_mode="human")] * environmentsNum, shared_memory=False)
        envs = gym.vector.AsyncVectorEnv([lambda: gym.make('SpaceInvaders-v0', render_mode="human")] * environmentsNum, shared_memory=False)


    randomActionFlag = 1
    randomActionNum = 10
    randomActionCounter = 0

    frameSkipFlag = 1
    frameSkipNum = 4
    frameSkipCounter = 0
    

    bufferStage = 0

    tarNetUpdateThresh = 5
    tarNetUpdateCounter = 0


    episodeStoreThresh = 10
    episodeStoreCounter = 0


    rewardDiscountFact = 1.0 

    epsClip = 0.2
    evalFact = 1.0
    entropyFact = 0.2
    ppoFact = 0

    paramClipHigh = 1.0
    paramClipLow = -1.0

    evalClipHigh = 0.5
    evalClipLow = -0.5

    evalLearnRate = -1.0
    evalDiscountFactor = 0.99

    batchGradEvalAcumNum = 1

    rewardDiscountAverage = 0.0

    evalErrorCountThresh = 10
    evalErrorCount = 0
    evalErrorThresh = 0.1
    evalErrorHitFlag = 0
    evalErrorList = []


    whileCounterLimit = 10
    whileCounter = 0

    policyGradTest = 1

    sigma = 0.1
    learning_rate = 0.001

    #rewardAveragingPeriod = 10
    rewardAveragingPeriod = 3

    rewardAvePeriodTemp = 0.0
    rewardAvePeriodCounter = 0

    rewardAvePeriodArrLen = int(numEpisodes / rewardAveragingPeriod)

    rewardRecentAverage = 0.0
    rewardEpisodeTermThresh = 300.0

    rewardGraphHeight = 350

    softmaxTestFlag = 1
    softmaxDerTestFlag = 1

    episodeEndEarlyFlag = 0
    hyperParamAveEpisodeList = np.zeros((numHyperParamTests, numAveragingRuns), dtype = int)
    hyperAveEpisodeList = np.zeros(numHyperParamTests, dtype = int)


    valuePointCounter = 1  
    valueErrorRmsList = []
    valueErrorPointList = []  
    valueErrorRmsMin = 0.1
    valueErrorRmsCurrent = 1.0

    valueErrorBreakFlag = 0  

    GradUpdateCounter = 0

    for hyperParamTest in range(numHyperParamTests):

       print('The hyper parameter test run is: ' + str(hyperParamTest + 1))
       print()

       rewardDiscountFact = rewardDiscountFactorList[hyperParamTest]

       evalDiscountFactor = evalDiscountFactorList[hyperParamTest] 

       evalFact = evalFactorList[hyperParamTest]
       entropyFact = entropyFactorList[hyperParamTest]
       ppoFact = ppoFactorList[hyperParamTest]

       evalLearnRate = evalLearnRateList[hyperParamTest] 

       batchGradAcumNum = batchGradAcumNumList[hyperParamTest]

       for aveRun in range(numAveragingRuns):

          print('The averaging run is: ' + str(aveRun + 1))
          print()

          learningRateQLearn = learningRateQList[hyperParamTest]
          learningRateQDec = learningRateQDecList[hyperParamTest]  

          optimizer = keras.optimizers.Adam(learning_rate=learningRateQLearn)


          if(loadPolicyFlag == 1):

              print('Policy Loading Stage hit.')
              print()
              
              policyModel1.load_weights('weights.h5')
              
              with open('optimizer.pkl', 'rb') as f:
                 weight_values = pickle.load(f)

              optimizer.set_weights(weight_values)

          allRewardArr = np.zeros(numEpisodes, dtype = float)
          aveRewardArr = np.zeros(numEpisodes, dtype = float)
          aveRewardMax = 0.0

          rewardRecentAverage = 0.0
          aveRewardMax = 0

          episodeEndEarlyFlag = 0

          tarNetUpdateCounter = 0  

          GradUpdateCounter = 0

          augTwinUpdateCounter = 0

          param1UpdateCounter = 0
          param2UpdateCounter = 0
          param3UpdateCounter = 0

          paramAugUpdateCounter = 0

          bufAdded = 0
          bufCounter = 0
          bufFinished = 0

          episodeStoreCounter = 0

          trainStageFlag = 0

          inValAllImageBuf = np.zeros((1, policyImageShape[0], policyImageShape[1], policyImageShape[2]), dtype = float)
          
          outValBuf = np.zeros((1, OL), dtype = float)

          evalValBuf = np.zeros((1, EL), dtype = float)
          actionBuf = np.zeros(1, dtype = float)
          rewardBuf = np.zeros(1, dtype = float)
          doneBuf = np.zeros(1, dtype = float)
          advanValBuf = np.zeros((1, EL), dtype = float)
          policyStartFinish = np.zeros(1, dtype = float)

          nextActionBatch = np.random.choice(OL, environmentsNum)

          rewardsBatch = np.random.choice(1, environmentsNum)

   

          inValAllImageBatchBuf = np.zeros((parallelDatastoreNum, stepLimit, policyImageShape[0], policyImageShape[1], policyImageShape[2]), dtype = float)

          outValBatchBuf = np.zeros((parallelDatastoreNum, stepLimit, OL), dtype = float)

          evalValBatchBuf = np.zeros((parallelDatastoreNum, stepLimit, EL), dtype = float)

          actionsBatchBuf = np.zeros((parallelDatastoreNum, stepLimit), dtype = float)
          donesBatchBuf = np.zeros((parallelDatastoreNum, stepLimit), dtype = float)

          rewardBatchBuf = np.zeros((parallelDatastoreNum, stepLimit), dtype = float)
          rewardRunningBatchBuf = np.zeros((parallelDatastoreNum), dtype = float)

          runFinishedBatch = np.zeros(parallelDatastoreNum, dtype=int)  
          startIndecesBatch = np.zeros(parallelDatastoreNum, dtype=int)
          endIndecesBatch = np.zeros(parallelDatastoreNum, dtype=int)
          difIndecesBatch = np.zeros(parallelDatastoreNum, dtype=int)

          runActualStepBatch = np.zeros(parallelDatastoreNum, dtype=int)
          runSkipStepBatch = np.zeros(parallelDatastoreNum, dtype=int)


          inValAllImageBatchFlatBuf = np.zeros((flattenedBatchNum, policyImageShape[0], policyImageShape[1], policyImageShape[2]), dtype = float)

          outValBatchFlatBuf = np.zeros((flattenedBatchNum, OL), dtype = float)

          evalValBatchFlatBuf = np.zeros((flattenedBatchNum, EL), dtype = float)

          actionsBatchFlatBuf = np.zeros(flattenedBatchNum, dtype = float)
          startBatchFlatBuf = np.zeros(flattenedBatchNum, dtype = int)
          donesBatchFlatBuf = np.zeros(flattenedBatchNum, dtype = int)

          rewardBatchFlatBuf = np.zeros(flattenedBatchNum, dtype = float)

          rewardAvePeriodList = []

          timeStart = timer()

          

          #start learning
          for episode in range(numEpisodes):

              learningRateQLearn -= learningRateQDec

              if((episode % 5) == 0):

                  timeCur = timer()
                  timeSoFar = (timeCur - timeStart)

                  print('The stats at episode: ' + str(episode) + ' with time to run so far: ' + str(timeSoFar) + ' are learning rate: ' + str(learningRateQLearn) + ' and recent average reward: ' + str(rewardRecentAverage))
                  print()

                  inValImageBatch[:, :, :, :] = 0.0
                  inValImageOld1Batch[:, :, :, :] = 0.0
                  inValImageOld2Batch[:, :, :, :] = 0.0
                  inValImageOld3Batch[:, :, :, :] = 0.0

                  inValAllImageBatch[:, :, :, :] = 0.0
 
                  evalValBatch[:] = 0.0

                  outValBatch[:] = 0.0

                  rewardsBatch[:] = 0.0
                  donesBatch[:] = 0.0

                  rewardRunningBatchBuf[:] = 0.0

                  runActualStepBatch[:] = 0
                  runSkipStepBatch[:] = 0

                  startBatchFlatBuf[:] = 0

                  startIndecesBatch[:] = 0
                  endIndecesBatch[:] = 0
                  difIndecesBatch[:] = 0

                  nextAction = 0
                  nextActionOld1 = 0
                  nextActionOld2 = 0                                                               
                  nextActionOld3 = 0     

                  observations = envs.reset()
                  observations = observations[0]

                  runStarted = 0
                  step = 0

                  randomActionCounter = 0
                  randomActionFlag = 1

                  frameSkipCounter = 0

                  bufferStage = 0

                  completedRunCounter = 0
                  
                  while True:
                      
                      inValImageOld3Batch[:] = inValImageOld2Batch[:]
                      inValImageOld2Batch[:] = inValImageOld1Batch[:]
                      inValImageOld1Batch[:] = inValImageBatch[:]
                      
                      inValImageBatch[:, :, :, :] = observations[:, :, :, :]
                      inValImageBatch[:] = ((inValImageBatch[:] / (np.max(inValImageBatch[:]) - np.min(inValImageBatch[:]))) - 0.5)

                      inValAllImageBatch[:, :, :, :3] = inValImageBatch[:]
                      inValAllImageBatch[:, :, :, 3:6] = inValImageOld1Batch[:]
                      inValAllImageBatch[:, :, :, 6:9] = inValImageOld2Batch[:]
                      inValAllImageBatch[:, :, :, 9:] = inValImageOld3Batch[:]   

                  
                      policyOutputBatch = policyModel1(inValAllImageBatch.reshape(environmentsNum, policyImageShape[0], policyImageShape[1], policyImageShape[2]))

                      evalValBatch = policyOutputBatch[0].numpy()

                      outValBatch = policyOutputBatch[1].numpy()


                      #Normal section
                      #nextActionBatch = np.random.choice(3, environmentsNum,  p = outValBatch)

                      for envIndex in range(environmentsNum):

                          nextActionBatch[envIndex] = np.random.choice(policyActionNum, 1,  p = outValBatch[envIndex,:])
                          inValAllImageBatchBuf[: envIndex, runSkipStepBatch[envIndex], :, :, :] = inValAllImageBatch[envIndex, :, :, :]
            
                      
                      observations, rewardsBatch, donesBatch, truncationsBatch, infosBatch = envs.step(nextActionBatch) 
   
                      outValBatchBuf[: environmentsNum, step, :] = outValBatch[:]

                      evalValBatchBuf[: environmentsNum, step, :] = evalValBatch[:]

                      actionsBatchBuf[: environmentsNum, step] = nextActionBatch[:]
                      rewardBatchBuf[: environmentsNum, step] = rewardsBatch[:]

                      donesBatchBuf[: environmentsNum, step] = np.where(donesBatch == True, 1, 0)


                      rewardRunningBatchBuf[: environmentsNum] += rewardsBatch[:]

                      runSkipStepBatch[:] += 1

                    

                      if(step >= (stepLimit - 1)):
                          
                          for envIndex in range(environmentsNum):

                              if(runFinishedBatch[envIndex] == 0):
                                  
                                  donesBatchBuf[envIndex, step] = 1
                                  donesBatch[envIndex] = 1

                          trainStageFlag = 1

                      for envIndex in range(environmentsNum):

                          stepLimitOffset = (envIndex * stepLimit) 

                          if(donesBatch[envIndex]):
                              startIndecesBatch[envIndex] = 0
                              endIndecesBatch[envIndex] = step

                              completedRunCounter += 1

                          inValAllImageBatchFlatBuf[(stepLimitOffset + step)] = inValAllImageBatchBuf[envIndex, step, :, :, :]
                          
                          outValBatchFlatBuf[(stepLimitOffset + step), :] = outValBatchBuf[envIndex, step, :] 
                          evalValBatchFlatBuf[(stepLimitOffset + step)] = evalValBatchBuf[envIndex, step]
                          
                          actionsBatchFlatBuf[(stepLimitOffset + step)] = actionsBatchBuf[envIndex, step]
                          donesBatchFlatBuf[(stepLimitOffset + step)] = donesBatchBuf[envIndex, step]
                          rewardBatchFlatBuf[(stepLimitOffset + step)] = rewardBatchBuf[envIndex, step]
                          
                          
                      
                      if(trainStageFlag == 1):

                          print('Batch training stage hit')
                          print()
    
                          policyOutput = policyModel1(inValAllImageBatchFlatBuf[inValAllImageBatchFlatBuf.shape[0] - 1].reshape(1, policyImageShape[0], policyImageShape[1], policyImageShape[2]))
                          next_value = policyOutput[0].numpy()[0].reshape(1, -1)

                          discounted_rewards, advantages = get_advantages(rewardBatchFlatBuf, donesBatchFlatBuf, evalValBatchFlatBuf, rewardDiscountFact)

                          actions = actionsBatchFlatBuf
                          probs = outValBatchFlatBuf
                          action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1)

                          total_loss = np.zeros((numTrainEpochs))
                          act_loss = np.zeros((numTrainEpochs))
                          c_loss = np.zeros(((numTrainEpochs)))
                          ent_loss = np.zeros((numTrainEpochs))

                         
                            
                          for epoch in range(numTrainEpochs):

                             trainSampledIndices = np.random.randint(0, actions.shape[0], ppoBatchSize)

                             

                             discountedRewardBatch = discounted_rewards[trainSampledIndices]
                             advantagesBatch = advantages[trainSampledIndices]

                             probsBatch = probs[trainSampledIndices]
                             actionsBatch = actions[trainSampledIndices]

                             ppoBatchRange = tf.range(0, ppoBatchSize)
                             actionsCasted = tf.cast(actionsBatch, tf.int32)

                            
                             actionIndBatch = tf.stack([tf.range(0, ppoBatchSize), tf.cast(actionsBatch, tf.int32)], axis=1)

                             inValAllImageTrainBatch = inValAllImageBatchFlatBuf[trainSampledIndices]
                              

                             loss_tuple = train_model(policyModel1, actionIndBatch, tf.gather_nd(probsBatch, actionIndBatch),
                                                      inValAllImageTrainBatch, advantagesBatch, discountedRewardBatch, optimizer,
                                                      criticLossWeight, ent_discount_val, ppoClipVal) 
                             
                             total_loss[epoch] = loss_tuple[0]
                             c_loss[epoch] = loss_tuple[1]
                             act_loss[epoch] = loss_tuple[2]
                             ent_loss[epoch] = loss_tuple[3]
                                 
                             ent_discount_val *= ent_discount_rate
            
                             policyGradTest = 1

                          inValAllImageBuf = np.zeros((1, policyImageShape[0], policyImageShape[1], policyImageShape[2]), dtype = float)

                          outValBuf = np.zeros((1, OL), dtype = float)
                          
                          evalValBuf = np.zeros((1, EL), dtype = float)
                          
                          actionBuf = np.zeros(1, dtype = float)
                          rewardBuf = np.zeros(1, dtype = float)
                          doneBuf = np.zeros(1, dtype = float)
                          policyStartFinish = np.zeros(1, dtype = float)


                          trainStageFlag = 0
                          

                      for envIndex in range(environmentsNum):

                          runActualStepBatch[envIndex] += 1
                      
                      step += 1
                      
                      if (step >= stepLimit):
                          break


              for envIndex in range(environmentsNum):

                  allRewardArr[episode] += rewardRunningBatchBuf[envIndex]

              allRewardArr[episode] /= environmentsNum
            
              if((episode >= (rewardAveragingPeriod - 1)) and (((episode + 1) % rewardAveragingPeriod) == 0)):  
     
                 rewardRecentAverage = np.mean(allRewardArr[(episode - rewardAveragingPeriod + 1): (episode + 1)])
                 
                 if(rewardRecentAverage >= rewardEpisodeTermThresh):

                    hyperParamAveEpisodeList[hyperParamTest, aveRun] = episode

                    episodeEndEarlyFlag = 1

                    print('The reward averaging period is: ')
                    print(rewardAveragingPeriod)
                    print()
                    print('The reward threshold is: ')
                    print(rewardEpisodeTermThresh)
                    print()
                    print('The reward recent average:')
                    print(rewardRecentAverage)
                    print()

                    print('The minimum reward threshhold has been reached')
                    print()
                 
                    break  
            
          if(episodeEndEarlyFlag == 0):

             hyperParamAveEpisodeList[hyperParamTest, aveRun] = numEpisodes
             
              
          #env.close()
          envs.close()

          timeEnd = timer()

          timeDif = (timeEnd - timeStart)

          print('The run has ended')
          print()

          print('The time taken for ' + str(hyperParamAveEpisodeList[hyperParamTest, aveRun]) +' episodes is: '+ str(timeDif))
          print()


          print('The gradient update number for this run is: ' + str(GradUpdateCounter))
          print()


          xPointsAve = []
          xPointsAll = []

          #xPnt = 1
          xPnt = 0

          rewardAvePeriodTemp = 0.0
          rewardAvePeriodCounter = 0

          print('The graph generation section is hit. ')
          print()
          print()
          print('The shape of allRewardArray is: ')
          print(allRewardArr.shape)
          print()
          
          for episode in range(numEpisodes):

              if((episode >= (rewardAveragingPeriod - 1)) and (((episode + 1) % rewardAveragingPeriod) == 0)): 

                 rewardRecentAverage = np.mean(allRewardArr[(episode - rewardAveragingPeriod + 1): (episode + 1)]) 
                

                 if(rewardRecentAverage >= aveRewardMax):
                     aveRewardMax = rewardRecentAverage

                 rewardAvePeriodList.append(rewardRecentAverage)

                 xPnt = episode   
                 xPointsAve.append(xPnt)

                 
          xPointsArrAve = np.array(xPointsAve)  
          rewardAvePeriodArr = np.array(rewardAvePeriodList)


          print('The reward average period array is: ')
          print(rewardAvePeriodArr)
          print()

          plt.axis([0, numEpisodes, 0, rewardGraphHeight])
          plt.xlabel('episodes')
          plt.ylabel('average reward')
          plt.title('plotted episodes and average reward values')
          plt.legend(['line 1'])

          plt.plot(xPointsArrAve, rewardAvePeriodArr)

          plt.savefig('./perform_plot_stats/ave_reward_plot_hyp_param_' + str(hyperParamTest + 1) + '_ave_run_' + str(aveRun + 1) + '_with_max_ave_reward_' + str(aveRewardMax) +  '.png', dpi=300, bbox_inches='tight')

          plt.close()


    hyperAveEpisodeList = np.mean(hyperParamAveEpisodeList, axis=1)

    print('The episode counts for each averaging run and hyper parameter run are: ')
    print(hyperParamAveEpisodeList[:, :])
    print()
    print('The average episode counts for each hyper parameter run are: ')
    print(hyperAveEpisodeList)
    print()

    if(savePolicyFlag == 1):

       print('Policy Saving Stage Hit')
       print()

       policyModel1.save_weights('weights.h5')
       symbolic_weights = getattr(policyModel1.optimizer, 'weights')
       weight_values = K.batch_get_value(symbolic_weights)
       with open('optimizer.pkl', 'wb') as f:
          pickle.dump(weight_values, f) 

