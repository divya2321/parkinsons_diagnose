#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2  
import os


# ## Creating Folder Structure

# In[2]:


#Creating Training, Validation and Testic Dataset folders
subfolder_names = ['Training Data', 'Validation Data', 'Testing Data']
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join('Dataset/Split Image/Data vertical 2', subfolder_name, 'Healthy'))
    os.makedirs(os.path.join('Dataset/Split Image/Data vertical 2', subfolder_name, 'PD'))


# In[3]:


#Creating Training, Validation and Testic Dataset folders
subfolder_names = ['Training Data', 'Validation Data', 'Testing Data']
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join('Dataset/Split Image/Data vertical 4', subfolder_name, 'Healthy'))
    os.makedirs(os.path.join('Dataset/Split Image/Data vertical 4', subfolder_name, 'PD'))


# In[4]:


#Creating Training, Validation and Testic Dataset folders
subfolder_names = ['Training Data', 'Validation Data', 'Testing Data']
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join('Dataset/Split Image/Data horizontal 2', subfolder_name, 'Healthy'))
    os.makedirs(os.path.join('Dataset/Split Image/Data horizontal 2', subfolder_name, 'PD'))


# In[5]:


#Creating Training, Validation and Testic Dataset folders
subfolder_names = ['Training Data', 'Validation Data', 'Testing Data']
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join('Dataset/Split Image/Data horizontal 4', subfolder_name, 'Healthy'))
    os.makedirs(os.path.join('Dataset/Split Image/Data horizontal 4', subfolder_name, 'PD'))


# --------------------------

# # Split Images Vertically

# ### Into 2

# In[6]:


train_healthy_images = os.listdir('Dataset/Full Image/Training Data/Healthy')

for file in train_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 2/Training Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 2/Training Data/Healthy/' + file + "_2.png", s2)


# In[7]:


val_healthy_images = os.listdir('Dataset/Full Image/Validation Data/Healthy')

for file in val_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 2/Validation Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 2/Validation Data/Healthy/' + file + "_2.png", s2)


# In[8]:


test_healthy_images = os.listdir('Dataset/Full Image/Testing Data/Healthy')

for file in test_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 2/Testing Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 2/Testing Data/Healthy/' + file + "_2.png", s2)


# In[9]:


train_pd_images = os.listdir('Dataset/Full Image/Training Data/PD')

for file in train_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 2/Training Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 2/Training Data/PD/' + file + "_2.png", s2)


# In[10]:


val_pd_images = os.listdir('Dataset/Full Image/Validation Data/PD')

for file in val_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 2/Validation Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 2/Validation Data/PD/' + file + "_2.png", s2)


# In[11]:


test_pd_images = os.listdir('Dataset/Full Image/Testing Data/PD')

for file in test_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 2/Testing Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 2/Testing Data/PD/' + file + "_2.png", s2)


# -----------

# ### Into 4

# In[12]:


train_healthy_images = os.listdir('Dataset/Full Image/Training Data/Healthy')

for file in train_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in quarter
    width_cutoff = width // 4
    s1 = img[:, :width_cutoff]
    s2 = img[:, :width_cutoff + width_cutoff]
    s2 = s2[:, width_cutoff:]
    s3 = img[:, width_cutoff + width_cutoff:]
    s3 = s3[:, :width_cutoff]
    s4 = img[:, width_cutoff + width_cutoff + width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/Healthy/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/Healthy/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/Healthy/' + file + "_4.png", s4)


# In[13]:


val_healthy_images = os.listdir('Dataset/Full Image/Validation Data/Healthy')

for file in val_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in quarter
    width_cutoff = width // 4
    s1 = img[:, :width_cutoff]
    s2 = img[:, :width_cutoff + width_cutoff]
    s2 = s2[:, width_cutoff:]
    s3 = img[:, width_cutoff + width_cutoff:]
    s3 = s3[:, :width_cutoff]
    s4 = img[:, width_cutoff + width_cutoff + width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/Healthy/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/Healthy/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/Healthy/' + file + "_4.png", s4)


# In[15]:


test_healthy_images = os.listdir('Dataset/Full Image/Testing Data/Healthy')

for file in test_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in quarter
    width_cutoff = width // 4
    s1 = img[:, :width_cutoff]
    s2 = img[:, :width_cutoff + width_cutoff]
    s2 = s2[:, width_cutoff:]
    s3 = img[:, width_cutoff + width_cutoff:]
    s3 = s3[:, :width_cutoff]
    s4 = img[:, width_cutoff + width_cutoff + width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/Healthy/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/Healthy/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/Healthy/' + file + "_4.png", s4)


# In[16]:


train_pd_images = os.listdir('Dataset/Full Image/Training Data/PD')

for file in train_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in quarter
    width_cutoff = width // 4
    s1 = img[:, :width_cutoff]
    s2 = img[:, :width_cutoff + width_cutoff]
    s2 = s2[:, width_cutoff:]
    s3 = img[:, width_cutoff + width_cutoff:]
    s3 = s3[:, :width_cutoff]
    s4 = img[:, width_cutoff + width_cutoff + width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/PD/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/PD/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Training Data/PD/' + file + "_4.png", s4)


# In[17]:


val_pd_images = os.listdir('Dataset/Full Image/Validation Data/PD')

for file in val_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in quarter
    width_cutoff = width // 4
    s1 = img[:, :width_cutoff]
    s2 = img[:, :width_cutoff + width_cutoff]
    s2 = s2[:, width_cutoff:]
    s3 = img[:, width_cutoff + width_cutoff:]
    s3 = s3[:, :width_cutoff]
    s4 = img[:, width_cutoff + width_cutoff + width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/PD/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/PD/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Validation Data/PD/' + file + "_4.png", s4)


# In[18]:


test_pd_images = os.listdir('Dataset/Full Image/Testing Data/PD')

for file in test_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in quarter
    width_cutoff = width // 4
    s1 = img[:, :width_cutoff]
    s2 = img[:, :width_cutoff + width_cutoff]
    s2 = s2[:, width_cutoff:]
    s3 = img[:, width_cutoff + width_cutoff:]
    s3 = s3[:, :width_cutoff]
    s4 = img[:, width_cutoff + width_cutoff + width_cutoff:]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/PD/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/PD/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data vertical 4/Testing Data/PD/' + file + "_4.png", s4)


# -------------------------------------------------

# --------------------------------------------------

# # Split Images Horizontally

# ### Into 2

# In[19]:


train_healthy_images = os.listdir('Dataset/Full Image/Training Data/Healthy')

for file in train_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 2
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Training Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Training Data/Healthy/' + file + "_2.png", s2)


# In[20]:


val_healthy_images = os.listdir('Dataset/Full Image/Validation Data/Healthy')

for file in val_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 2
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Validation Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Validation Data/Healthy/' + file + "_2.png", s2)


# In[21]:


test_healthy_images = os.listdir('Dataset/Full Image/Testing Data/Healthy')

for file in test_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 2
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Testing Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Testing Data/Healthy/' + file + "_2.png", s2)


# In[22]:


train_pd_images = os.listdir('Dataset/Full Image/Training Data/PD')

for file in train_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 2
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Training Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Training Data/PD/' + file + "_2.png", s2)


# In[25]:


val_pd_images = os.listdir('Dataset/Full Image/Validation Data/PD')

for file in val_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 2
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Validation Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Validation Data/PD/' + file + "_2.png", s2)


# In[26]:


test_pd_images = os.listdir('Dataset/Full Image/Testing Data/PD')

for file in test_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 2
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Testing Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 2/Testing Data/PD/' + file + "_2.png", s2)


# ----------------------------------------------------

# ### Into 4

# In[27]:


train_healthy_images = os.listdir('Dataset/Full Image/Training Data/Healthy')

for file in train_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 4
    s1 = img[:height_cutoff, :]
    s2 = img[:height_cutoff+height_cutoff, :]
    s2 = s2[height_cutoff:, :]
    s3 = img[height_cutoff+height_cutoff:, :]
    s3 = s3[:height_cutoff, :]
    s4 = img[height_cutoff+height_cutoff+height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/Healthy/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/Healthy/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/Healthy/' + file + "_4.png", s4)


# In[28]:


val_healthy_images = os.listdir('Dataset/Full Image/Validation Data/Healthy')

for file in val_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 4
    s1 = img[:height_cutoff, :]
    s2 = img[:height_cutoff+height_cutoff, :]
    s2 = s2[height_cutoff:, :]
    s3 = img[height_cutoff+height_cutoff:, :]
    s3 = s3[:height_cutoff, :]
    s4 = img[height_cutoff+height_cutoff+height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/Healthy/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/Healthy/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/Healthy/' + file + "_4.png", s4)


# In[29]:


test_healthy_images = os.listdir('Dataset/Full Image/Testing Data/Healthy')

for file in test_healthy_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/Healthy/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 4
    s1 = img[:height_cutoff, :]
    s2 = img[:height_cutoff+height_cutoff, :]
    s2 = s2[height_cutoff:, :]
    s3 = img[height_cutoff+height_cutoff:, :]
    s3 = s3[:height_cutoff, :]
    s4 = img[height_cutoff+height_cutoff+height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/Healthy/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/Healthy/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/Healthy/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/Healthy/' + file + "_4.png", s4)


# In[31]:


train_pd_images = os.listdir('Dataset/Full Image/Training Data/PD')

for file in train_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Training Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 4
    s1 = img[:height_cutoff, :]
    s2 = img[:height_cutoff+height_cutoff, :]
    s2 = s2[height_cutoff:, :]
    s3 = img[height_cutoff+height_cutoff:, :]
    s3 = s3[:height_cutoff, :]
    s4 = img[height_cutoff+height_cutoff+height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/PD/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/PD/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Training Data/PD/' + file + "_4.png", s4)


# In[32]:


val_pd_images = os.listdir('Dataset/Full Image/Validation Data/PD')

for file in val_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Validation Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 4
    s1 = img[:height_cutoff, :]
    s2 = img[:height_cutoff+height_cutoff, :]
    s2 = s2[height_cutoff:, :]
    s3 = img[height_cutoff+height_cutoff:, :]
    s3 = s3[:height_cutoff, :]
    s4 = img[height_cutoff+height_cutoff+height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/PD/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/PD/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Validation Data/PD/' + file + "_4.png", s4)


# In[ ]:


test_pd_images = os.listdir('Dataset/Full Image/Testing Data/PD')

for file in test_pd_images:
    # Read the image
    img = cv2.imread('Dataset/Full Image/Testing Data/PD/' + file)
    height = img.shape[0]
    width = img.shape[1]

    # Cut the image in half
    height_cutoff = height // 4
    s1 = img[:height_cutoff, :]
    s2 = img[:height_cutoff+height_cutoff, :]
    s2 = s2[height_cutoff:, :]
    s3 = img[height_cutoff+height_cutoff:, :]
    s3 = s3[:height_cutoff, :]
    s4 = img[height_cutoff+height_cutoff+height_cutoff:, :]
    
    file = file.split('.png')[0]

    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/PD/' + file + "_1.png", s1)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/PD/' + file + "_2.png", s2)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/PD/' + file + "_3.png", s3)
    cv2.imwrite('Dataset/Split Image/Data horizontal 4/Testing Data/PD/' + file + "_4.png", s4)


# In[ ]:





# In[ ]:




