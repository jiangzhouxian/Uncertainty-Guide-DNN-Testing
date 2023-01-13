#Calculation of various uncertainty metrics：PCS、VR、VRO、PE、PE-DP、MI、Conflict、Ignorancce、non-specifycity,Ensemble,Ensemble-dp
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras import backend as K
#Calculate the metric PCS, which returns the pcs of all samples
import heapq
def threshold_PCS(x, model):
    y = model.predict(x)
    a = list()
    b = list()
    for i in range(len(y)):
        a.append(heapq.nlargest(2,y[i]))
    a = np.array(a)
    for j in range(len(y)):
        b.append(abs(a[j][0]-a[j][1]))
    b = np.array(b)
    return b    
#pcs = threshold_PCS(X_adv,pre_model)
#print(pcs)

#The calculation of the metric VR, which returns the VR of a single sample
def threshold_VR(x,model_dp,T):
    T_label = []
    for k in range(T):
        y_pre = model_dp.predict(x.reshape(-1,28,28,1))[0]
        y_pre_max = np.argmax(y_pre) # k-th prediction label
        T_label.append(y_pre_max)
    T_label = np.array(T_label)
    T_times = np.argmax(np.bincount(T_label))
    count = np.bincount(T_label)[T_times]
    return 1-(count/T)

#The calculation of the metric VRO, which returns the VRO of a single sample
def threshold_VRO(x, model, model_dp,T):  
    #orig_model prediction result
    y_pre = model.predict(x.reshape(-1,28,28,1))[0]
    y_lab = np.argmax(y_pre)
    count = 0
    for i in range(T):
        #采样T次
        y_pre_dp = model_dp.predict(x.reshape(-1,28,28,1))[0]
        y_pre_dp_max = np.argmax(y_pre_dp)
        if y_pre_dp_max == y_lab:
            count += 1
    return 1-(count/T)


#PE
def threshold_PE(x, model,n_classes):
    pd = 0
    pe = 0
    y_pre_dp = model.predict(x.reshape(-1,28,28,1))[0]
    for i in range(n_classes):
        pe += -y_pre_dp[i]*np.log(y_pre_dp[i])
    return pe

#PE-DP and MI
def threshold_MI(x, model, T ,n_classes):
    pd = []
    pe = 0
    pp = 0
    mi = 0
    for j in range(T):
        y_pre_dp = model.predict(x.reshape(-1,28,28,1))[0]
        pd.append(y_pre_dp)
    pd = np.array(pd)
    pd = np.sum(pd, axis=0)
    
    for i in range(n_classes):
        pe += -((1/T)*pd[i])*np.log((1/T)*pd[i])
        pp += pd[i]*np.log(pd[i])
    mi = pe + (1/T)*pp
    return pe,mi

#conflict/ignorance/non_specifycity
import NNbeliefFunc as nnbelief

def threshold_DS(x,model,no):
    if no == 0:
        outputs = [layer.output for layer in model.layers[6:]]        # all layer outputs
        functor = K.function([model.input],outputs)
        fc_outputs = functor([x])
        fc1_outputs = fc_outputs[0]
        fc2_outputs = fc_outputs[1]
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[4] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[5] # bias of the 1st hidden layer（fc1—fc2）
    elif no == 1:
        outputs = [layer.output for layer in model.layers[8:]]        # all layer outputs
        functor = K.function([model.input],outputs)
        fc_outputs = functor([x])
        fc1_outputs = fc_outputs[0]
        fc2_outputs = fc_outputs[1]
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[6] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[7] # bias of the 1st hidden layer（fc1—fc2）
    elif no == 2: 
        outputs = [layer.output for layer in model.layers[10:]]        # all layer outputs
        functor = K.function([model.input],outputs)
        fc_outputs = functor([x])
        fc1_outputs = fc_outputs[0]
        fc2_outputs = fc_outputs[1]
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[8] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[9] # bias of the 1st hidden layer（fc1—fc2）
    elif no == 3:
        outputs = [layer.output for layer in model.layers[20:]]        # all layer outputs
        functor = K.function([model.input],outputs)
        fc_outputs = functor([x])
        fc1_outputs = fc_outputs[0]
        fc2_outputs = fc_outputs[1]
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[12] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[13] # bias of the 1st hidden layer（fc1—fc2）
          
    _beta_0_output =nnbelief.beta_0(beta_0_output)
    _beta_output=nnbelief.beta_jk(beta_output)
    _a_output =nnbelief.alpha_jk(_beta_output,_beta_0_output,fc2_outputs)
    k=nnbelief.conflict(_beta_output,_a_output,fc2_outputs) 
    ig =nnbelief.ignorance(_beta_output,_a_output,fc2_outputs)
    u_set = np.arange(0,_a_output.shape[1]) 
    non_sp = nnbelief.H_non_specificity(_beta_output,_a_output,fc2_outputs,u_set)
    return k,ig,non_sp


#En
def threshold_Ensemble(x,model,Models):
    Y = []
    #求对模型model的真实标签
    lab = np.argmax(model.predict(x.reshape(-1,28,28,1))[0])
    for md in Models:
        y_pre = md.predict(x.reshape(-1,28,28,1))[0]
        Y.append(y_pre)
    Y = np.array(Y)
    var = np.var(Y, axis = 0)[lab]
    
    return var

# En-DP
def threshold_Ensemble_dp(x,model,Models,T):
    pre = []
    for i in range(T):
        pre.append(model.predict(x.reshape(-1,28,28,1))[0])
    pre = np.mean(pre,axis = 0)
    lab = np.argmax(pre)
    
    Var = []    
    for model in Models:
        
        Y_pre = []
        for i in range(T):
            y_pre = model.predict(x.reshape(-1,28,28,1))[0]
            Y_pre.append(y_pre)
        Y_pre = np.array(Y_pre)
        y_mean = np.mean(Y_pre,axis = 0)
        #print('y_mean = ',y_mean)
        Var.append(y_mean)
    Var = np.array(Var)
    final_var = np.var(Var,axis = 0)[lab]
    return final_var