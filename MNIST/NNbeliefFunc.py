#!/usr/bin/env python
# coding: utf-8

#  This is a python implementation of the work by [Denoeux (2019)](https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/publi/nnbelief_kbs_v2_clean.pdf) and the entropy calculation w.r.t. mass functions defined by [Jiroušek and Shenoy (2018)](http://library.utia.cas.cz/separaty/2017/MTR/jirousek-0481470.pdf)

# In[2]:


import numpy as np


# ## Parameter determination

# $\beta ^*_{0k}=\hat{\beta_{0k}}-\frac{1}{K}\sum^K_{l=1}\hat{\beta_{0l}} \quad$

# In[2]:


def beta_0(b): # Equ.38 ("_beta" -> ""beta^*"")
    return b-np.mean(b)


# Beta: $\beta ^*_{jk}=\hat{\beta_{jk}}-\frac{1}{K}\sum^K_{l=1}\hat{\beta_{jl}} \quad$ (Equ. B.3)

# In[3]:


def beta_jk(w):
    beta_temp = np.sum(w,axis =1)/w.shape[1]
    beta_temp = beta_temp.reshape(len(beta_temp),-1)
    beta_temp = np.tile(beta_temp,(1,w.shape[1]))
    beta_temp = w - beta_temp
    return beta_temp


# Alpha: $\alpha^*_{jk}=\frac{1}{J}\left(\beta^*_{0k}+\sum^J_{j=1}\beta^*_{jk}\mu_j \right)-\beta_{jk}^* \mu_j$   (Equ.38)

# In[4]:


# Mean output for each neuron in a layer
def mean_neuron(output):
    u_temp = np.mean(output,axis=0)
    return u_temp # tensor shape(j,)

def alpha_jk_single(beta,beta_0): # possibily incorrect, not from the original paper
    alpha_temp = beta_0/beta.shape[0]
    alpha_temp = np.tile(alpha_temp,(beta.shape[0],1))
    alpha_temp.shape
    return alpha_temp

def alpha_jk(beta,beta_0,x):
    beta_temp = beta_jk(beta)
    u_temp = mean_neuron(x)
    _u_temp =u_temp.reshape(len(u_temp),-1)
    _u_temp = np.tile(_u_temp,(1,beta.shape[1]))
    temp=(beta_0+np.dot(beta_temp.T,u_temp))/len(u_temp)
    alpha_temp = np.zeros([beta.shape[0],beta.shape[1]]) #tensor shape(j,k)
    for i in range(len(beta_temp)):
        alpha_temp[i]= temp -(beta_temp*_u_temp)[i]
    return alpha_temp


# ## Prediction information derived based on NNbelief 

# $w_{jk}:=\beta_{jk}\phi_j(x)+\alpha_{jk}   $  (Equ. 25)

# In[5]:


def belief_weight(beta,alpha,x):
    weight_jk=np.zeros([len(x),len(beta),len(beta[0])])
    weight_temp=np.zeros([len(beta),len(beta[0])])
    for i in range(len(weight_jk)):
        for j in range(beta.shape[1]):
            weight_temp[:,j]=beta[:,j]*x[i] 
        weight_jk[i]=weight_temp+alpha 
    return weight_jk


# $w_k^+ := \sum_{j=1}^J w_{jk}^+$  (Equ. 27)

# In[6]:


def weight_pos(beta,alpha,x):
    weight = belief_weight(beta,alpha,x)
    weight_temp=np.zeros([len(weight),len(weight[0][0])])  # tensor shape(n,k)
    for i in range(len(weight)):
        weight_temp[i]=np.maximum(0,weight[i]).sum(axis=0)
    return weight_temp


# $w_k^- := \sum_{j=1}^J w_{jk}^-$  (Equ. 27)

# In[7]:


def weight_neg(beta,alpha,x):
    weight = belief_weight(beta,alpha,x)
    weight_temp=np.zeros([len(weight),len(weight[0][0])])  # tensor shape(n,k)
    for i in range(len(weight)):
        weight_temp[i]=np.maximum(0,-weight[i]).sum(axis=0)
    return weight_temp


# $\eta^+ = (\sum_{l=1}^K exp(w_l^+)-K+1)^{-1}$ (Equ. A.3.-1)

# In[8]:


def eta_pos(beta,alpha,x):
    w_pos = weight_pos(beta,alpha,x)
    eta_temp=np.zeros([len(w_pos)]) # tensor shape(n,)
    K=len(w_pos[0])
    for i in range(len(w_pos)):
        eta_temp[i]=1/(np.exp(w_pos[i]).sum()-K+1)
    return eta_temp


# $\eta^- = (1-\prod_{l=1}^K[1-exp(-w_l^-)])^{-1}$ (Equ. A.3.-2)

# In[9]:


def eta_neg(beta,alpha,x):
    w_neg = weight_neg(beta,alpha,x)
    eta_temp=np.zeros([len(w_neg)]) # tensor shape(n,)
    for i in range(len(w_neg)):
        eta_temp[i]=1/(1-np.prod(1-(np.exp(-w_neg[i])+1e-15)))
    #print(eta_temp)
    return eta_temp


# **Conflict** 
# 
# $k=\sum_{k=1}^K\{\eta^+(exp(w_k^+)-1)[1-\eta^-exp(-w_k^-)]\}$(Equ. A.3.-3)

# In[10]:


def conflict(beta,alpha,x):
    w_pos = weight_pos(beta,alpha,x)
    w_neg = weight_neg(beta,alpha,x)
    k_temp=np.zeros([len(w_neg)])
    eta_pos_temp = eta_pos(beta,alpha,x)
    eta_neg_temp = eta_neg(beta,alpha,x)
    #print(eta_pos_temp)
    #print(eta_neg_temp)
    for i in range(len(w_neg)):
        k_temp[i]=sum(eta_pos_temp[i]*(np.exp(w_pos[i])-1)*(1-eta_neg_temp[i]*np.exp(-w_neg[i])))
    return k_temp


# $\eta = (1-k)^{-1}$(Equ. A.3.-4)

# In[11]:


def eta(beta,alpha,x):
    return 1/(1-(conflict(beta,alpha,x)-(1e-15)))


# **Ignorance** 
# 
# $m(\Theta) = \eta\cdot\eta^+\cdot\eta^-\cdot \exp(-\sum_{k=1}^K w_k^-) $

# In[12]:


def ignorance(beta,alpha,x):
    w_pos = weight_pos(beta,alpha,x)
    w_neg = weight_neg(beta,alpha,x)
    eta_pos_temp = eta_pos(beta,alpha,x)
    eta_neg_temp = eta_neg(beta,alpha,x)
    eta_temp = eta(beta,alpha,x)
    ig_temp = eta_temp*eta_pos_temp*eta_neg_temp*np.exp(-np.sum(w_neg,axis=1))
    return ig_temp


# **Mass($m(\theta_k)$)**
# 
# $m(\theta_k) = \eta\cdot\eta^+\cdot\eta^-\cdot \exp(-w_k^-)\cdot \{ \exp(w_k^+)-1+\prod_{l\neq k}[1-\exp(-w_l^-)]\}$(Equ. A.3.-6)

# In[13]:


def m_theta_k(beta,alpha,x): # return the masses for all k
    w_pos = weight_pos(beta,alpha,x)
    w_neg = weight_neg(beta,alpha,x)
    eta_pos_temp = eta_pos(beta,alpha,x)
    eta_neg_temp = eta_neg(beta,alpha,x)
    eta_temp = eta(beta,alpha,x)        
    prod_temp = np.zeros([len(x)])
    m_temp = np.zeros([len(x),len(w_neg[0])])
    for i in range(len(w_neg[0])):
        w_neg_cut = np.delete(w_neg,i,1)
        for j in range(len(w_neg_cut)):
            prod_temp[j] = np.prod(1-np.exp(-w_neg_cut[j]))
        m_temp[:,i] = eta_temp*eta_pos_temp*eta_neg_temp*np.exp(-w_neg[:,i])*(np.exp(w_pos[:,i])-1+prod_temp)
    return m_temp


# **Mass($m(A)$)**
# 
# $m(A) = \eta\cdot\eta^+\cdot\eta^-\cdot \{\prod_{\theta_k\notin A}[1-\exp(-w_k^-)]\}\cdot \{\prod_{\theta_k\in A}\exp(-w_k^-)\}$(Equ. A.3.-7)

# In[ ]:


def m_theta_12(beta,alpah,x): 
    w_pos = weight_pos(beta,alpah,x)
    w_neg = weight_neg(beta,alpah,x)
    eta_pos_temp = eta_pos(beta,alpah,x)
    eta_neg_temp = eta_neg(beta,alpah,x)
    eta_temp = eta(beta,alpah,x)
    return(eta_temp*eta_pos_temp*eta_neg_temp*(1-np.exp(-w_neg[:,2]))*np.exp(-w_neg[:,0])*np.exp(-w_neg[:,1]))


# In[14]:


def m_theta_A(beta,alpha,x,A):
    u_set = np.arange(0,alpha.shape[1]) # Universal set
    
    idx_A = []
    for i in A:
        idx_A.append(np.where(u_set == i)[0][0]) # get the index of A
    idx_notA = list(np.delete(u_set,idx_A)) # get the index of not A
    
    w_pos = weight_pos(beta,alpha,x)
    w_neg = weight_neg(beta,alpha,x)
    eta_pos_temp = eta_pos(beta,alpha,x)
    eta_neg_temp = eta_neg(beta,alpha,x)
    eta_temp = eta(beta,alpha,x)
    
    w=np.prod(np.exp(-w_neg[:,idx_A]),axis=1)
    _w=np.prod(1-np.exp(-w_neg[:,idx_notA]),axis=1)
    
    return(eta_temp*eta_pos_temp*eta_neg_temp*w*_w)


# In[15]:


# calculate powerset without empty set
# https://blog.csdn.net/xjtuse123/article/details/99202846
def PowerSetsBinary(items):
    N = len(items)
    arr = []
    for i in range(2 ** N): # Number of subsets
        combo = []
        for j in range(N): 
            if(i>>j)%2: # check if the bit whose index equals to j is 1
                combo.append(items[j])
        arr.append(combo)
    arr.remove([]) # remove empty set
    return arr

a=PowerSetsBinary([4,5,6])
a


# In[3]:


# Complment of A 补集
def complement(u_set,A):
    idx_A = []
    if(len(A)!=1):
        for i in A:
            idx_A.append(np.where(u_set == i)[0][0])
    else:
        idx_A.append(np.where(u_set == A)[0])
    idx_notA = list(np.delete(u_set,idx_A))
    return(u_set[idx_notA])


# **Belief function**
# 
# $bel(A) = \sum_{B\subseteq A}m(B)$(Equ. 2a)

# In[17]:


def bel(beta,alpha,x,u_set,A):
    sub = PowerSetsBinary(A)
    bel_temp = np.zeros([len(x)])
    for i in sub:
        if(len(i)!=1): # use m_A when the subset has more than one element
            bel_temp=bel_temp+m_theta_A(beta,alpha,x,i)
        else: # use m_k when the subset has only one element
            bel_temp=bel_temp+m_theta_k(beta,alpha,x)[:,i[0]]
    return(bel_temp)


# **Plausibility function**
# 
# $pl(A) = \sum_{B\cap A \neq \emptyset}m(B) = 1-bel(\bar{A})$(Equ. 2b)

# In[18]:


def plausibility(beta,alpha,x,u_set,A):
    com_A = complement(u_set,A)
    pl = 1-np.round(bel(beta,alpha,x,u_set,com_A),14) # round number to avoid computational error(NAN)
    return(pl)


# ## Calculate the entropy

# **Shannon's entropy with plausibility transform**
# 
# $H_s(P_m)=\sum_{x\in \Omega_X}P_m(x)log\big(\frac{1}{P_m(x)}\big)$

# In[19]:


def H_conflict(beta,alpha,x,u_set):
    # plausibility transform (Voorbraak,1989; Shenoy,2006)
    sum_pl = np.zeros([len(x)]) # normalization constant for plausibility transform
    p_m = []
    for i in u_set:
        i=[i]
        sum_pl = sum_pl+plausibility(beta,alpha,x,u_set,i)
    for j in u_set:
        j=[j]
        p_m.append(plausibility(beta,alpha,x,u_set,j)/sum_pl) #(Equ. 9)
    p_m = np.array(p_m)
    # Shannon's entropy
    h_temp = (p_m*np.log2(1/p_m)).sum(axis=0)
    return(h_temp)


# # **Dubois and Prade's non-specificity entropy (Dubois and Prada, 1987)**
# 
# $H_d(m)=\sum_{a\in 2^{\Omega_X}}m(a)log(|a|)$

# In[20]:


def H_non_specificity(beta,alpha,x,u_set):
    p_set = PowerSetsBinary(u_set)
    h_temp = []
    for i in p_set:
        if(len(i)>1):
            h_temp.append(m_theta_A(beta,alpha,x,i)*np.log2(len(i)))
    h_temp = np.array(h_temp)            
    h_temp = h_temp.sum(axis=0)
    return(h_temp)


# **Total entropy: conflict + non-specificity (Shenoy,2018)**
# 
# $H(m)=H_s(P_m)+H_d(m)=\sum_{x\in \Omega_X}P_m(x)log\big(\frac{1}{P_m(x)}\big)+\sum_{a\in 2^{\Omega_X}}m(a)log(|a|)$

# In[21]:


def H_total(beta,alpha,x,u_set):  
    return(H_conflict(beta,alpha,x,u_set)+H_non_specificity(beta,alpha,x,u_set))


# ##  Calculate the decision  interval 

#  **lower expected loss**   
#  $R_*(a_k)=1-pl(\theta_k)$ (Equ. 2.17-a)

# In[ ]:


def low_loss(beta,alpha,x,u_set,A):
    ra=1-plausibility(beta,alpha,x,u_set,A)
    return (ra)


#  **upper expected loss**    
#  $R^*(a_k)=1-Bel(\theta_k)$ (Equ. 2.17-b)

# In[ ]:


def up_loss(beta,alpha,x,u_set,A):
    ra=1-bel(beta,alpha,x,u_set,A)
    return (ra)


# In[ ]:




