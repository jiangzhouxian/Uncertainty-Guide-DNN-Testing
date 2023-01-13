from __future__ import print_function
import tensorflow as tf

import argparse
#from keras.datasets import mnist
from keras.layers import Input
from keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from scipy.io import loadmat
import ModelA
import ModelB
import ModelC
import ModelE
from utils import *
import imageio
import numpy as np
import math
import time
import datetime
import Metrics as mc
random.seed(3)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for uncertainty guided input generation in SVHN dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['no','central_crop','random_crop', 'flip','brightness','contrast'])
parser.add_argument('decay', help="decay fator to control mim goal", type=float, default = 1.0)
parser.add_argument('eps', help="the max perturbation", type=float, default = 8/255)
parser.add_argument('alpha', help="the step of perturbation", type=float, default = 2/255)
parser.add_argument('weight_mim', help="weight hyperparm to control loss goal", type=float)
parser.add_argument('weight_uc', help="weight hyperparm to control uncertainty goal", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2, 3], default=0, type=int)
args = parser.parse_args()

print("\n\n")
output_directory = './0507UC_shift/Model' + str(args.target_model + 1)+'/'+ str(args.transformation)+'/'
Valid_dir = output_directory+'Valid/'

#Create directory to store generated tests
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
delete_files_from_dir(output_directory, 'png')

if not os.path.exists(Valid_dir):
    os.makedirs(Valid_dir)
delete_files_from_dir(Valid_dir, 'png')

# Create a subdirectory inside output directory 
# for saving original seed images used for test generation
orig_directory = output_directory+'seeds/'
orig_valid_dir = Valid_dir+'seeds/'

if not os.path.exists(orig_directory):
    os.makedirs(orig_directory)
delete_files_from_dir(orig_directory, 'png')

if not os.path.exists(orig_valid_dir):
    os.makedirs(orig_valid_dir)
delete_files_from_dir(orig_valid_dir, 'png')


# input image dimensions
img_rows, img_cols = 32, 32
img_chn = 3
img_dim = 1000
# Load the test data
datasetLoc = './dataset/'
test_data = loadmat(datasetLoc+'test_32x32.mat')
x_test = np.array(test_data['X'])
y_test = test_data['y']

# Normalize data.
x_test = np.moveaxis(x_test, -1, 0)
x_test = x_test.astype('float32') / 255

y_test[y_test == 10] = 0
y_test = np.array(y_test)
    
input_shape = (img_rows, img_cols, img_chn)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)

modelA = ModelA.ModelA(input_tensor=input_tensor)
modelB = ModelB.ModelB(input_tensor=input_tensor)
modelC = ModelC.ModelC(input_tensor=input_tensor)
modelE = ModelE.ModelE(input_tensor=input_tensor)
Models = [modelA,modelB,modelC,modelE]
Models_na = ['modelA','modelB','modelC','modelE']

print("*****Running Uncertainty test....")

start_time = datetime.datetime.now()
seed_nums = np.load('./seeds/seeds_'+str(args.seeds)+'.npy')
result_loop_index = []
#result_coverage = []

total_l2=0
total_perturb_adversial=0
no_tests=0
no_valid=0
seed = 0
actual_l2 = 0
for current_seed in seed_nums:
    iter = 0
    # below logic is to track number of iterations under progress
    seed += 1
    loop_index = 0
    orig_img = np.expand_dims(x_test[current_seed], axis=0)
    before_shift = orig_img
    
    if args.transformation =='central_crop':
        orig_img = tf.image.central_crop(orig_img,0.9)
        orig_img = tf.image.resize_with_crop_or_pad(orig_img,32,32)
        orig_img = orig_img.numpy()
        
               
    elif args.transformation =='flip':
        orig_img = tf.image.random_flip_left_right(orig_img)
        orig_img = orig_img.numpy()
    
    elif args.transformation =='random_crop':
        orig_img = tf.image.resize_with_crop_or_pad(orig_img,31,31)
        orig_img = tf.image.resize_with_crop_or_pad(orig_img,32,32)
        orig_img = orig_img.numpy() 
                
    elif args.transformation =='brightness':
        orig_img =tf.image.random_brightness(orig_img,0.005)
        orig_img = orig_img.numpy()
    elif args.transformation =='contrast':
        orig_img =tf.image.random_contrast(orig_img,0.00001,0.00005)
        orig_img = orig_img.numpy()
       
    gen_img = orig_img.copy()
    if args.target_model == 0:
        pre_model =  modelA
        thre_PE_dp = 1.0923
        thre_En = 0.0613
        thre_En_dp = 0.0188
    elif args.target_model == 1:
        pre_model =  modelB
        thre_PE_dp = 1.0923
        thre_En = 0.0613
        thre_En_dp = 0.0188

    elif args.target_model == 2:
        pre_model =  modelC
        thre_PE_dp = 0.9565
        thre_En = 0.0657
        thre_En_dp = 0.0209

    elif args.target_model == 3:
        pre_model =  modelE
        thre_PE_dp = 0.9028
        thre_En = 0.0264
        thre_En_dp = 0.0454
        
    #原始样本的ground true标签
    orig_label = y_test[current_seed]
    orig_label = orig_label[0]
    #cross_entropy_loss
    y_true = orig_label
    orig_label = to_categorical(orig_label,num_classes = 10)
   
    loss_ce = tf.nn.softmax_cross_entropy_with_logits(logits = pre_model.output,labels = orig_label)               
    grad_ce = K.gradients(loss_ce,input_tensor)

    #Uncertainty_entropy
    outputs = pre_model.output
    loss_en = -K.sum(K.softmax(outputs,axis=1)*K.log(K.softmax(outputs,axis=1)+(1e-8)),axis = 1)
    grad_en = K.gradients(loss_en,input_tensor)
    
     
    # function to calculate the loss and grads given the input tensor
    iterate = K.function([input_tensor], [grad_ce,grad_en])

    # Running gradient ascent
    g = np.zeros(shape=(32,32,3)) 
    img_list =[]
    img_list.append(gen_img)
    
    while len(img_list)>0:
        gen_img = img_list[0]
        img_list.remove(gen_img)
        
        for iters in range(3):
            iter +=1
            if np.linalg.norm(gen_img-orig_img)>3 or iter>args.grad_iterations:
                break
            else:
                loop_index +=1
                grad_ce,grad_en = iterate([gen_img])
                grad_ce = grad_ce[0]
                grad_en = grad_en[0]
                
                grad_ce,grad_en = np.array(grad_ce),np.array(grad_en)
                    #print(grad_ce.shape,grad_en.shape)
                total_grad = args.weight_mim*grad_ce-args.weight_uc*grad_en
                    #print(total_grad,total_grad.shape)
                total_grad = total_grad / K.mean(K.abs(total_grad), axis=(1,2,3))
                    #print(total_grad,total_grad.shape)

                    #更新g
                g = g*args.decay + total_grad
                    #print(g)
                    #更新x
                per = args.alpha * np.sign(g)
                diff = np.clip(per,-args.eps,args.eps)
                gen_img = np.clip(gen_img + diff, 0,1)
                pred1 = np.argmax(pre_model.predict(gen_img)[0])

                    #绝对l2距离
                diff_img = gen_img-orig_img
                l2_norm = np.linalg.norm(diff_img)
                    #相对l2距离
                orig_l2 = np.linalg.norm(orig_img)
                perturb_adversial = l2_norm/orig_l2
                img_list.append(gen_img)
                #en = mc.threshold_Ensemble(gen_img,pre_model,Models)
                #ensemble = mc.threshold_Ensemble(gen_img,pre_model,Models)
                Pe_dp,_ = mc.threshold_MI(gen_img, pre_model, 10 ,10)
                #en_dp = mc.threshold_Ensemble_dp(gen_img,pre_model,Models,10)

                if not pred1 == y_true:
                    if Pe_dp < thre_PE_dp:
                        #print('Is valid!')
                        #print('The l2 and perturb_adversial distance of generated img are:',l2_norm,perturb_adversial)
                        total_l2+=l2_norm
                        total_perturb_adversial+=perturb_adversial
                        actual_l2 += np.linalg.norm(gen_img-before_shift)
                        result_loop_index.append(loop_index)
                        no_valid +=1
                        gen_img_tmp = gen_img.copy()

                        gen_img_deprocessed = deprocess_image(gen_img_tmp)
                        orig_img_deprocessed = deprocess_image(orig_img)
                            # save the result to disk
                        imageio.imwrite(Valid_dir+'_' +str(seed)+'_' +str(loop_index) + '_' 
                                                + str(pred1) + '_' + str(y_true) +'.png',
                                                gen_img_deprocessed)
                        imageio.imwrite(orig_valid_dir+'_' +str(seed)+'_' +str(loop_index) + '_' 
                                                + str(pred1) + '_' + str(y_true)+'_orig.png',
                                                orig_img_deprocessed)
                break
    
duration = (datetime.datetime.now() - start_time).total_seconds()
#no_tests = len(result_loop_index)

print('model,trans',args.target_model,args.transformation)
print("**** Result of uncertainty test:")
#print("No of test inputs generated: ", no_tests)
print("No of valid test inputs generated: ", no_valid)
if no_valid == 0:
    print("Cumulative coverage for tests: 0")
    print('Avg. test generation time: NA s')
else:
    #print("Cumulative coverage for tests: ", round(result_coverage[-1],3))
    print('Avg. test l2 distance: {} and {}'.format(total_l2/no_valid,actual_l2/no_valid))
    #print('valid test generation: {} %'.format(100*(no_valid/no_tests)))
    print('Avg. test generation time: {} s'.format(round(duration/no_valid),2))
print('Total time: {} s'.format(round(duration, 2)))
