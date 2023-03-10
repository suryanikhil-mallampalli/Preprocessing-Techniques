# -*- coding: utf-8 -*-
"""DS_Project_Dataset_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cx1gKb4gRvc9LyRVhk_3HHZ6f4zyymyJ

#Importing

Importing Required Modules
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

"""Loading Data """

df=pd.read_csv('/content/Salaries.csv')

"""#Preprocessing

**Normalization**<br>

Min-Max Scaler Applied
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
TotalPayReshaped = df.TotalPay.to_numpy().reshape(-1, 1)
df.TotalPay = scaler.fit_transform(TotalPayReshaped)
df.hist(figsize=(15,6), column=['TotalPay'])

"""**Outlier removal**<br>
 an outlier is an observation point that is distant from other observations.
"""

x = df.TotalPay
UPPERBOUND, LOWERBOUND = np.percentile(x, [1,99])
y = np.clip(x, UPPERBOUND, LOWERBOUND)
pd.DataFrame(y).hist()

"""**Log Transformation**<br>
The log transformation is another commonly used technique when you want to reduce the variability of data. Another popular use of the log transformation is when the data distribution is highly skewed.
"""

df['LogTotalPay'] = np.log(1+df.TotalPay)
df.describe()