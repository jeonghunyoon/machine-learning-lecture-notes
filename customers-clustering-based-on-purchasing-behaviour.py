# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Complete Source code with Explanation: https://github.com/Balakishan77/Clustering-of-Customers-Based-on-their-purchasing-behaviour
#Clustering of customers based on Number of items Purchased(Quantity),Product price per unit in sterling(Unit Price)'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print (os.listdir("../input"))
# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/data.csv",encoding = "ISO-8859-1") # Encoded as this dataset contains West Europe countries retail transactions
dataset.shape  #(541909, 8)
'''
> dataset.select_dtypes(include=['object']).columns
Index(['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate', 'Country'], dtype='object')
> dataset.dtypes
InvoiceNo       object
StockCode       object
Description     object
Quantity         int64
InvoiceDate     object
UnitPrice      float64
CustomerID     float64
Country         objectvnv
dtype: object
> dataset['Country'].dtype
dtype('O')
'''
#will remove the duplicate entries in the datset - 5268
print (dataset.duplicated().sum())    
dataset.drop_duplicates(inplace = True)
dataset.shape   #(536641, 8)
#Removing missing values based on  CustomerID.
dataset.dropna(axis = 0, subset =['CustomerID'], inplace = True)
dataset.shape #(406829, 8)
print (pd.DataFrame(dataset.isnull().sum()))    #Checking for any null entries column wise, We can see that there are 0 null entries
#Removing Cancelled orders
dataset = dataset[(dataset.InvoiceNo).apply(lambda x:( 'C' not in x))]
dataset.shape    #(392732, 8)
df_customerid_groups=dataset.groupby("CustomerID")
print (len((df_customerid_groups.groups))) #length of dictionary - 4339
'''Creating a new dataframe with 'Quantity','UnitPrice','CustomerID' columns and we are adding unitprice and quantity 
in a grop of user, so will end up one row per one user'''
df_cluster=pd.DataFrame(columns=['Quantity','UnitPrice','CustomerID'])
count=0
for k,v in (df_customerid_groups):
    df_cluster.loc[count] = [(v['Quantity'].sum()), v['UnitPrice'].sum(), k]
    count+=1
df_cluster.shape  #(4339, 3)
# Applying K-Means Clustering Algorithm
# We use only 'Quantity','UnitPrice' columns to cluser 
X = df_cluster.iloc[:, [0, 1]].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X= sc_X.fit_transform(X)
#Using the Elbow method to find the optical number of clusters
from sklearn.cluster import KMeans
wcss = [] #With in cluster sum of squers(Inertia)
'''
#n_clusters is no.of clusters given by this method,
#k-means++ is an random initialization methods for centriods to avoid random intialization trap,
#max_iter is max no of iterations defined when k-means is running
#n_init is no of times k-means will run with different initial centroids
'''
for i in range(1,11): #From 2-10 doing multiple random initializations can make a huge difference to find a better local optima
    kmeans = KMeans(n_clusters = i, init ='k-means++',max_iter=300,n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11) , wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Customer Clusters(customer type clusters)')
plt.ylabel('With in cluster sum of squers(WCSS)')
plt.show()
'''
From the plot we can see that at 3 distortion goes rapidly so n_clusters=3
'''
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Customer Type 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Customer Type 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Customer Type 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Type Of Customers(customer type clusters)')
plt.xlabel('Number of items Purchased(Quantity)')
plt.ylabel('Product price per unit in sterling(Unit Price)')
plt.legend()
plt.show()

x=[];y=[]
for i in range(4339):
    x.append(X[i][0])
    y.append(X[i][1])
plt.scatter(x,y)
plt.title('Plot of training data')
plt.xlabel('Number of items Purchased(Quantity)')
plt.ylabel('Product price per unit in sterling(Unit Price)')
plt.show()

