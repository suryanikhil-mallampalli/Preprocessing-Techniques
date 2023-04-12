# Data Science Project
<p> This project contains code for data preprocessing and analysis. There are two Jupyter notebooks, each containing code for a separate dataset. The purpose of this project is to demonstrate how to clean, preprocess, and visualize data.</p>

<h1> Dataset 1</h1>
<p>The first dataset is a salary dataset. The notebook DS_Project_Dataset_1.ipynb contains code to load, preprocess, and analyze the dataset.</p>

<pre>
<code>import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)</code></pre>
<h1> Loading Data</h1>
<p>The data is loaded into a Pandas DataFrame using the following code:</p>
<pre><code>df=pd.read_csv('/content/Salaries.csv')</code></pre>

<h1>Preprocessing</h1>
The following preprocessing steps are applied to the dataset:
<ul>
<li> Normalization: Min-Max Scaler is applied to normalize the `<b>TotalPay</b>` column.</li>
<li> Outlier removal: Outliers in the `<b>TotalPay</b>` column are removed using percentile-based clipping.</li>
<li> Log transformation: The `<b>TotalPay</b>` column is log-transformed to reduce data variability.</li>
</ul>

<h1>Analyzing</h1>
<p> Descriptive statistics are used to analyze the data, including the <b>`describe()`</b> method to summarize the dataset.</p>

<h1> Dataset 2</h1>
<p>he second dataset is a credit card dataset. The notebook <b>`DS_Project_Dataset_2.ipynb`</b> contains code to load, preprocess, and analyze the dataset.</p>

<h2>Importing Required Modules</h2>
The required modules are imported using the following code:
<pre><code>import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA</code></pre>
<h2> Loading Data</h2>
The data is loaded into a Pandas DataFrame using the following code:
<pre><code>df = pd.read_csv('/content/c_card.csv')</code></pre>
<h2>Preprocessing</h2>
The following preprocessing steps are applied to the dataset:
<ul>
<li>Checking for missing values.</li>
<li>Checking the distribution of classes.</li>
<li>Checking for categorical data.</li>
<li>Standardizing the features.</li>
<li> Applying PCA for dimensionality reduction.</li>
</ul>
<h2>Analyzing</h2>
Principal Component Analysis (PCA) is used to visualize the data in two dimensions.
<h1> Conclusion</h1>
This project demonstrates how to clean, preprocess, and visualize data using Python and various libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn.




