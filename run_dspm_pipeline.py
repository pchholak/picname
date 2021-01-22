#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from main_dspm_function import run_dspm
import numpy as np


# In[ ]:


# iSubjects = np.delete(np.arange(1, 16), [0, 1])
iSubjects = [3]


# In[ ]:


for iSub in iSubjects:
    print("Running dSPM for sub-%02d" % iSub)
    run_dspm(iSub, 'right')


# In[ ]:
