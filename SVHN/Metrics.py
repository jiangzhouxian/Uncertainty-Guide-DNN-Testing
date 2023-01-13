#各类不确定性指标的计算：PCS、VR、VRO、PE、MI、Conflict、Ignorancce、non-specifycity
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras import backend as K
#计算指标PCS，返回的是所有样本的pcs
import heapq
def threshold_PCS(x, model):
    y = model.predict(x)
    a = list()
    b = list()
    for i in range(len(y)):
        #把前2个最大值加入a中
        a.append(heapq.nlargest(2,y[i]))
    a = np.array(a)
    for j in range(len(y)):
        #求预测的最大值和次大值的差
        b.append(abs(a[j][0]-a[j][1]))
    b = np.array(b)
    return b    
#pcs = threshold_PCS(X_adv,pre_model)
#print(pcs)

#指标VR的计算，返回的是单个样本的vr
def threshold_VR(x,model_dp,T):
    T_label = []
    for k in range(T):
        y_pre = model_dp.predict(x.reshape(-1,32,32,3))[0]
        y_pre_max = np.argmax(y_pre) # k-th prediction label
        T_label.append(y_pre_max)
    T_label = np.array(T_label)
    T_times = np.argmax(np.bincount(T_label))
    count = np.bincount(T_label)[T_times]
    return 1-(count/T)

#指标VRO的计算，返回的是单个样本的vro
def threshold_VRO(x, model, model_dp,T):  
    #orig_model prediction result
    y_pre = model.predict(x.reshape(-1,32,32,3))[0]
    y_lab = np.argmax(y_pre)
    count = 0
    for i in range(T):
        #采样T次
        y_pre_dp = model_dp.predict(x.reshape(-1,32,32,3))[0]
        y_pre_dp_max = np.argmax(y_pre_dp)
        if y_pre_dp_max == y_lab:
            count += 1
    return 1-(count/T)

def threshold_PE(x, model,n_classes):
    pd = 0
    pe = 0
    y_pre_dp = model.predict(x.reshape(-1,32,32,3))[0]
    for i in range(n_classes):
        pe += -y_pre_dp[i]*np.log(y_pre_dp[i])
    return pe

    #指标MI的计算,返回的是单个样本的mi
def threshold_MI(x, model, T ,n_classes):
    pd = []
    pe = 0
    pp = 0
    mi = 0
    for j in range(T):
        y_pre_dp = model.predict(x.reshape(-1,32,32,3))[0]
        pd.append(y_pre_dp)
    pd = np.array(pd)
    pd = np.sum(pd, axis=0) #按列求和得到每一类的pd
    
    for i in range(n_classes):
        pe += -((1/T)*pd[i])*np.log((1/T)*pd[i]+1e-8)
        pp += pd[i]*np.log(pd[i]+1e-8)
    mi = pe + (1/T)*pp
    return pe,mi

#指标conflict/ignorance/non_specifycity的计算
import NNbeliefFunc as nnbelief

def threshold_DS(x,model,no):
    if no == 0:
        outputs = model.get_layer('global_average_pooling2d_1').output       # all layer outputs
        functor = K.function([model.input],[outputs])
    
      #提取output的输出（预测输出-概率值）
        #pred_outputs = fc_outputs[2]
        fc2_outputs = functor([x])
        beta_output=model.get_weights()[12] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[13] # bias of the 1st hidden layer（fc1—fc2）
    elif no == 1:
        outputs = [layer.output for layer in model.layers[19:]]        # all layer outputs
        functor = K.function([model.input],outputs)
      #获取各层的输出（X：输入样本）
        fc_outputs = functor([x])
      #提取FC1的输出
        fc1_outputs = fc_outputs[0]
      #提取FC2的输出
        fc2_outputs = fc_outputs[1]
      #提取output的输出（预测输出-概率值）
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[16] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[17] # bias of the 1st hidden layer（fc1—fc2）
    elif no == 2: 
        outputs = [layer.output for layer in model.layers[19:]]        # all layer outputs
        functor = K.function([model.input],outputs)
      #获取各层的输出（X：输入样本）
        fc_outputs = functor([x])
      #提取FC1的输出
        fc1_outputs = fc_outputs[0]
      #提取FC2的输出
        fc2_outputs = fc_outputs[1]
      #提取output的输出（预测输出-概率值）
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[16] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[17] # bias of the 1st hidden layer（fc1—fc2）
    elif no == 3:
        outputs = [layer.output for layer in model.layers[41:]]        # all layer outputs
        functor = K.function([model.input],outputs)
      #获取各层的输出（X：输入样本）
        fc_outputs = functor([x])
      #提取FC1的输出
        fc1_outputs = fc_outputs[0]
      #提取FC2的输出
        fc2_outputs = fc_outputs[1]
      #提取output的输出（预测输出-概率值）
        pred_outputs = fc_outputs[2]
        beta_output=model.get_weights()[36] # transposed weights of the 3st hidden layer（fc2—output）
        beta_0_output=model.get_weights()[37] # bias of the 1st hidden layer（fc1—fc2）
          
    _beta_0_output =nnbelief.beta_0(beta_0_output)
    _beta_output=nnbelief.beta_jk(beta_output)
    _a_output =nnbelief.alpha_jk(_beta_output,_beta_0_output,fc2_outputs)
    k=nnbelief.conflict(_beta_output,_a_output,fc2_outputs) 
    ig =nnbelief.ignorance(_beta_output,_a_output,fc2_outputs)
    #u_set = np.arange(0,_a_output.shape[1]) 
    #non_sp = nnbelief.H_non_specificity(_beta_output,_a_output,fc2_outputs,u_set)
    return k,ig

def threshold_Ensemble(x,model,Models):
    Y = []
    #求对模型model的真实标签
    lab = np.argmax(model.predict(x.reshape(-1,32,32,3))[0])
    for md in Models:
        y_pre = md.predict(x.reshape(-1,32,32,3))[0]
        Y.append(y_pre)
    Y = np.array(Y)
    var = np.var(Y, axis = 0)[lab]
    
    return var

def threshold_Ensemble_dp(x,model,Models,T):
    pre = []
    for i in range(T):
        pre.append(model.predict(x.reshape(-1,32,32,3))[0])
    pre = np.mean(pre,axis = 0)
    lab = np.argmax(pre)
    
    Var = []    
    for model in Models:
        
        Y_pre = []
        for i in range(T):
            y_pre = model.predict(x.reshape(-1,32,32,3))[0]
            Y_pre.append(y_pre)
        Y_pre = np.array(Y_pre)
        y_mean = np.mean(Y_pre,axis = 0)
        #print('y_mean = ',y_mean)
        Var.append(y_mean)
    Var = np.array(Var)
    final_var = np.var(Var,axis = 0)[lab]
    return final_var