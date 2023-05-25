#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import mne
import pywt as pywt
import os
import tensorflow as tf 
import random
import shutil
import gc

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from glob import glob
from autoreject import get_rejection_threshold
from random import shuffle
from tensorflow.keras import layers, models


# In[2]:


#EEG of healthy participants
hc_raw = glob("hc/*.bdf")

#EEG of PD patient's on their medication off
pd_raw = glob("pd/ses_off/*.bdf")


# In[3]:


def loaddata(data_file):
    #Retieve a sample EEG signal for one person
    return mne.io.read_raw_bdf(data_file, preload=True)  


# In[4]:


#Excluded the noisy channels and updated the channel list
ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']
    
#Override the info of the filtered sample
new_info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)

    
#Getting the digitized points of the head
montage_kind = "standard_1020"
montage =  mne.channels.make_standard_montage(montage_kind)


# In[5]:


#pre-process method
def preprocessdata(raw_file):
    #Filter the sample
    filtered_raw = raw_file.filter(l_freq = 0.8, h_freq = 30)
    filtered_raw.info = new_info
    filtered_raw.set_montage(montage, match_case=False)
    
    return filtered_raw


# In[6]:


#artifact removing method
def removeartifacts(filtered_raw_file):
    #Make fixed length events
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(filtered_raw_file, duration=tstep)
    epochs_ica = mne.Epochs(filtered_raw_file, events_ica,
                        tmin=0.0, 
                        tmax=tstep,
                        baseline=None,
                        preload=True)
    
    reject = get_rejection_threshold(epochs_ica);
    
    random_state = 42  
    ica_n_components = .99    

    ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state,)
    ica.fit(epochs_ica, reject=reject, tstep=tstep)
    
    ica_thresh = 1.96 
    eog_indices, eog_scores = ica.find_bads_eog(filtered_raw_file, ch_name=['Fp1', 'Fp2'], threshold=ica_thresh)
    ica.exclude = eog_indices
    
    return ica.apply(filtered_raw_file);


#  

# ------------------------------------------

# In[7]:


#Creating Training, Validation and Testic Dataset folders
subfolder_names = ['Training Data', 'Validation Data', 'Testing Data']
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join('Data', subfolder_name, 'Healthy'))
    os.makedirs(os.path.join('Data', subfolder_name, 'PD'))


# In[8]:


#splittinf epochs of the pre-processed files
def splitting_epochs(preprocessed_file):
    return mne.make_fixed_length_epochs(preprocessed_file, duration=5, preload=False)  


# In[9]:


#reshape the spochs by flattening
def reshape_epochs(epoch_files):
    
    epochs_array = []
    
    for epoch in epoch_files:
        
        for e in epoch.get_data():
            oneD = e.flatten()
            reshape_epochs = np.reshape(oneD, (-1,1024))
              
            for r in reshape_epochs:
                epochs_array.append(r)
                
    return epochs_array                 


# In[10]:


#creating datasets
healthy_training_data = []
healthy_validation_data = []
healthy_testing_data = []

pd_training_data = []
pd_validation_data = []
pd_testing_data = []

scales = np.arange(1, 33)

def create_datasets(reshaped_epochs, epoch_type):
    
    global healthy_training_data
    global healthy_validation_data
    global healthy_testing_data
    
    global pd_training_data
    global pd_validation_data
    global pd_testing_data
    
    
    epoch_count = 1
    
    epoch_size = int(len(reshaped_epochs))
    
    training_size = int(epoch_size*80/100)
    test_val_size = int(epoch_size*10/100)    
    validation_size = training_size + test_val_size
    
    
    if epoch_type == 'h': 
        
        for epoch in reshaped_epochs:
            
            if epoch_count <= training_size:
                healthy_training_data.append(epoch)
                epoch_count += 1
                
            elif epoch_count <= validation_size:
                healthy_validation_data.append(epoch)
                epoch_count += 1
                
            else:
                healthy_testing_data.append(epoch)
                epoch_count += 1
                
    else:
        
        for epoch in reshaped_epochs:
        
            if epoch_count <= training_size:
                pd_training_data.append(epoch)
                epoch_count += 1
                
            elif epoch_count <= validation_size:
                pd_validation_data.append(epoch)
                epoch_count += 1
                
            else:
                pd_testing_data.append(epoch)
                epoch_count += 1


# In[11]:


#generating the scalogram images and save
scales = np.arange(1, 33)

def convert_to_scalogram(recieved_sample, sample_status, sample_type):
    
    sample_no = 1
    
    for epoch in recieved_sample:

        coef, freqs = pywt.cwt(epoch, scales, 'gaus2') 
        
        plt.figure(figsize=(8, 5))
        plt.imshow(abs(coef), extent=[0, 1024, 33, 1], interpolation='bilinear', cmap='plasma',
                    aspect='auto', vmax=abs(coef).max(), vmin=abs(coef).min())
        plt.gca().invert_yaxis()
        plt.yticks(ticks=None, labels=None, minor=False)
        plt.xticks(ticks=None, labels=None, minor=False)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
    
        if sample_status == 'h':
            if sample_type == 'training':
                plt.savefig('Data/Training Data/Healthy/h_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
                
            elif sample_type == 'validation':
                plt.savefig('Data/Validation Data/Healthy/h_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
            
            elif sample_type == 'testing':
                plt.savefig('Data/Testing Data/Healthy/h_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
                
        else:
            if sample_type == 'training':
                plt.savefig('Data/Training Data/PD/p_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
                
            elif sample_type == 'validation':
                plt.savefig('Data/Validation Data/PD/p_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
            
            elif sample_type == 'testing':
                plt.savefig('Data/Testing Data/PD/p_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
            


# In[12]:


get_ipython().run_cell_magic('capture', '', 'healthy_read_files = [loaddata(r) for r in hc_raw]\npd_read_files = [loaddata(r) for r in pd_raw]\n')


# In[13]:


get_ipython().run_cell_magic('capture', '', 'healthy_preprocessed_files = [preprocessdata(r) for r in healthy_read_files]\npd_preprocessed_files = [preprocessdata(r) for r in pd_read_files]\n')


# In[14]:


get_ipython().run_cell_magic('capture', '', 'healthy_no_art_file = [removeartifacts(p) for p in healthy_preprocessed_files]\npd_no_art_file = [removeartifacts(p) for p in pd_preprocessed_files]\n')


# In[15]:


get_ipython().run_cell_magic('capture', '', 'healthy_epoch_files = [splitting_epochs(n) for n in healthy_no_art_file]\n')


# In[16]:


get_ipython().run_cell_magic('capture', '', 'pd_epoch_files = [splitting_epochs(n) for n in pd_no_art_file]\n')


# In[17]:


get_ipython().run_cell_magic('capture', '', 'healthy_reshape_epochs = reshape_epochs(healthy_epoch_files) \npd_reshape_epochs = reshape_epochs(pd_epoch_files) \n')


# In[18]:


random.shuffle(healthy_reshape_epochs)
random.shuffle(pd_reshape_epochs)


# In[19]:


selected_healthy_epochs = random.sample(healthy_reshape_epochs,5000)
selected_pd_epochs = random.sample(pd_reshape_epochs,5000)


# In[20]:


get_ipython().run_cell_magic('capture', '', "create_datasets(selected_healthy_epochs, 'h')\ncreate_datasets(selected_pd_epochs, 'p')\n")


# In[22]:


gc.collect()


# In[23]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(healthy_training_data, 'h', 'training')\n")


# In[24]:


gc.collect()


# In[25]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(pd_training_data, 'p', 'training')\n")


# In[26]:


gc.collect()


# In[27]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(healthy_validation_data, 'h', 'validation')\n")


# In[28]:


gc.collect()


# In[29]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(pd_validation_data, 'p', 'validation')\n")


# In[30]:


gc.collect()


# In[31]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(healthy_testing_data, 'h', 'testing')\n")


# In[32]:


gc.collect()


# In[33]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(pd_testing_data, 'p', 'testing')\n")


# In[ ]:




