#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn sqlalchemy')


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install SQLAlchemy')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sqlalchemy import create_engine


# In[4]:


get_ipython().system('pip install openpyxl')




# In[5]:


# Load the dataset
data = pd.read_csv('OnlineRetail.csv')  # Use read_csv for CSV files

# Display the first few rows
print(data.head())


# In[6]:


# Check the shape of the dataset
print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

# Display column names and data types
print(data.info())

# Summary statistics
print(data.describe())


# In[7]:


# Check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Drop rows with missing CustomerID or other essential fields
data = data.dropna(subset=['CustomerID'])

# Optionally, drop rows with negative or zero Quantity or UnitPrice
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]


# In[8]:


# Select relevant columns
data = data[['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Country']]

# Display remaining columns
print(data.columns)


# In[9]:


# Load the dataset
data = pd.read_csv('OnlineRetail.csv')  # or the correct file path

# Check the DataFrame structure and column names
print(data.head())
print(data.columns)

# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Create Total Spend feature
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

# Define the reference date as the day after the last invoice date
reference_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)

# Check if 'InvoiceNo' exists in the DataFrame
if 'InvoiceNo' in data.columns:
    # RFM Calculation
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).reset_index()

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSpend': 'Monetary'
    }, inplace=True)

    print(rfm.head())
else:
    print("Column 'InvoiceNo' not found in the DataFrame.")


# In[10]:


# Define a function to remove outliers based on IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Remove outliers for Monetary
rfm = remove_outliers(rfm, 'Monetary')

# Similarly, you can remove outliers for Recency and Frequency if needed


# In[11]:


# Features to scale
features = ['Recency', 'Frequency', 'Monetary']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
rfm_scaled = scaler.fit_transform(rfm[features])

# Convert to DataFrame
rfm_scaled = pd.DataFrame(rfm_scaled, columns=features)

print(rfm_scaled.head())


# In[12]:


# Recency
plt.figure(figsize=(10,4))
sns.histplot(rfm['Recency'], bins=30, kde=True)
plt.title('Recency Distribution')
plt.show()

# Frequency
plt.figure(figsize=(10,4))
sns.histplot(rfm['Frequency'], bins=30, kde=True)
plt.title('Frequency Distribution')
plt.show()

# Monetary
plt.figure(figsize=(10,4))
sns.histplot(rfm['Monetary'], bins=30, kde=True)
plt.title('Monetary Distribution')
plt.show()


# In[13]:


plt.figure(figsize=(8,6))
sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[14]:


# Average Order Value
rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']

# Lifetime
data_sorted = data.sort_values(['CustomerID', 'InvoiceDate'])
first_purchase = data_sorted.groupby('CustomerID')['InvoiceDate'].min().reset_index()
last_purchase = data_sorted.groupby('CustomerID')['InvoiceDate'].max().reset_index()
lifetime = pd.merge(first_purchase, last_purchase, on='CustomerID')
lifetime['Lifetime'] = (lifetime['InvoiceDate_y'] - lifetime['InvoiceDate_x']).dt.days

rfm = pd.merge(rfm, lifetime[['CustomerID', 'Lifetime']], on='CustomerID')

# Scale additional features if needed
features_extended = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Lifetime']
scaler_extended = StandardScaler()
rfm_scaled_extended = scaler_extended.fit_transform(rfm[features_extended])
rfm_scaled_extended = pd.DataFrame(rfm_scaled_extended, columns=features_extended)

print(rfm_scaled_extended.head())


# In[15]:


# Elbow Method
sse = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse[k] = kmeans.inertia_

plt.figure(figsize=(10,6))
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()

# Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    score = silhouette_score(rfm_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10,6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Various k")
plt.show()


# In[16]:


# Initialize KMeans with optimal number of clusters
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(rfm_scaled)

# Assign clusters to the original dataframe
rfm['Cluster'] = kmeans.labels_

# Display the number of customers in each cluster
print(rfm['Cluster'].value_counts())


# In[17]:


sns.pairplot(rfm, vars=['Recency', 'Frequency', 'Monetary'], hue='Cluster', palette='viridis')
plt.show()


# In[18]:


cluster_profiles = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'AOV': 'mean',
    'Lifetime': 'mean'
}).round(1)

print(cluster_profiles)


# In[19]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'], 
           c=rfm['Cluster'], cmap='viridis', marker='o', alpha=0.6)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title('3D Scatter Plot of Customer Segments')
plt.show()


# In[20]:


# Create a connection to SQLite database (it will create the database file if it doesn't exist)
engine = create_engine('sqlite:///ecommerce_customer_segmentation.db')

# Store the RFM dataframe with cluster labels into a SQL table
rfm.to_sql('customer_segments', engine, if_exists='replace', index=False)

print("Customer segments have been successfully stored in the database.")


# In[21]:


# Example: Retrieve all customers in Cluster 0
cluster_0 = pd.read_sql_query("SELECT * FROM customer_segments WHERE Cluster = 0", engine)
print(cluster_0.head())

# Example: Get the number of customers in each cluster
cluster_counts = pd.read_sql_query("SELECT Cluster, COUNT(*) as Count FROM customer_segments GROUP BY Cluster", engine)
print(cluster_counts)


# In[ ]:




