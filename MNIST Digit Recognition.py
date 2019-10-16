#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from fastai.vision import *
from fastai.metrics import *


# In[6]:


path = untar_data(URLs.MNIST)
path


# In[7]:


path.ls()[0].ls()[1].ls()


# In[8]:


doc(ImageDataBunch)


# In[9]:


tfms=get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path,valid_pct=0.2, ds_tfms=tfms, size=24, bs=64)


# In[10]:


data.show_batch(rows=5, figsize=(8,8))


# In[11]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.fit(2)


# In[12]:


interp = ClassificationInterpretation.from_learner(learn)


# In[14]:


loses, index = interp.top_losses()


# In[19]:


interp.plot_top_losses(30, figsize=(15,15))


# In[20]:


interp.plot_confusion_matrix()


# In[22]:


interp.most_confused(10)


# In[ ]:




