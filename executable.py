import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import os

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
# st.sidebar.add_rows("This website includes following functionalities",
# "Data Pre Processing using Pandas", "Correcting","Completing",
# "Creating","Modeling using Sklearn")
st.sidebar.header("This website includes following functionalities:")
st.sidebar.button("Data Preprocessing")
st.sidebar.button("Data Analysis")
st.sidebar.button("Data Clustering")
st.sidebar.button("Cluster Exports")


st.title("K-Means Clustering on Customer Models")
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("Filename:", uploaded_file.name)
     with open(os.path.join(".",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())
     df = pd.read_csv(uploaded_file.name)

     #Wrting first few elements of the data
     st.header("Data Preprocessing")

     st.subheader("First five elements of the file.")
     
     st.dataframe(df.head())
     
     df.drop_duplicates(subset= None, keep = 'first', inplace = False)

     df.drop(columns=["CustomerID"], inplace = True)
     st.caption("Dropped the duplicate values.")


     st.subheader("Dimensions of the dataset.")
     st.write(df.shape)
     st.header("Data Analysis")
     st.subheader("Box Plots")
     plt.figure(figsize=(20,5))
     plt.subplot(1,2,1)
     sns.boxplot(y=df["Spending Score"], color="green")
     plt.subplot(1,2,2)
     sns.boxplot(y=df["Annual Income"], color = "red")
     plt.savefig("1")
     st.image("1.png")


     st.subheader("Bar Plots")
     genders = df.Gender.value_counts()
     sns.set_style("whitegrid")
     plt.figure(figsize=(20,5))
     sns.barplot(x=genders.index, y=genders.values)
     st.pyplot(plt)
     plt.savefig("2")

     st.subheader("Age Group Distribution Plot")
     age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
     age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
     age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
     age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
     age55above = df.Age[df.Age >= 56]

     x = ["18-25","26-35","36-45","46-55","55+"]
     y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

     plt.figure(figsize=(20,5))
     sns.barplot(x=x, y=y, palette="Paired")
     plt.title("Age Group vs Number of Customers")
     plt.xlabel("Age")
     plt.ylabel("Number of Customer")
     st.pyplot(plt)
     plt.savefig("3")

     st.subheader("Spending Score Distribution Plot")
     spsc1_20 = df["Spending Score"][(df["Spending Score"] >= 1) & (df["Spending Score"] <= 20)]
     spsc21_40 = df["Spending Score"][(df["Spending Score"] >= 21) & (df["Spending Score"] <= 40)]
     spsc41_60 = df["Spending Score"][(df["Spending Score"] >= 41) & (df["Spending Score"] <= 60)]
     spsc61_80 = df["Spending Score"][(df["Spending Score"] >= 61) & (df["Spending Score"] <= 80)]
     spsc81_100 = df["Spending Score"][(df["Spending Score"] >= 81) & (df["Spending Score"] <= 100)]

     spscx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
     spscy = [len(spsc1_20.values), len(spsc21_40.values), len(spsc41_60.values), len(spsc61_80.values), len(spsc81_100.values)]

     plt.figure(figsize=(20,5))
     sns.barplot(x=spscx, y=spscy, palette="Set2")
     plt.title("Spending Scores")
     plt.xlabel("Score")
     plt.ylabel("Spending Score Group vs Number of Customers")
     st.pyplot(plt)
     plt.savefig("4")

     st.subheader("Annual Income Distribution Plot")
     ai0_30 = df["Annual Income"][(df["Annual Income"] >= 0) & (df["Annual Income"] <= 30)]
     ai31_60 = df["Annual Income"][(df["Annual Income"] >= 31) & (df["Annual Income"] <= 60)]
     ai61_90 = df["Annual Income"][(df["Annual Income"] >= 61) & (df["Annual Income"] <= 90)]
     ai91_120 = df["Annual Income"][(df["Annual Income"] >= 91) & (df["Annual Income"] <= 120)]
     ai121_150 = df["Annual Income"][(df["Annual Income"] >= 121) & (df["Annual Income"] <= 150)]

     aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
     aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

     plt.figure(figsize=(20,5))
     sns.barplot(x=aix, y=aiy, palette="nipy_spectral_r")
     plt.title("Annual Incomes")
     plt.xlabel("Income")
     plt.ylabel("Number of Customer")
     st.pyplot(plt)
     plt.savefig("5")

     st.header("Data Clustering using K-Means Clustering")
     st.subheader("Finding the value of K")

     inertia = []
     for k in range(1,21):
         kmeans = KMeans(n_clusters=k, init="k-means++")
         kmeans.fit(df.iloc[:,1:])
         inertia.append(kmeans.inertia_)
     
     plt.figure(figsize=(20,5))    
     plt.grid()
     plt.plot(range(1,21),inertia, linewidth=2, color="blue", marker ="8")
     plt.xlabel("K Value")
     plt.xticks(np.arange(1,21,1))
     plt.ylabel("Within Cluster Sum of Squares")
     st.pyplot(plt)

     st.write("Optimal K is equal to 5.")

     st.subheader("Clustering the Data")
     km = KMeans(n_clusters=5)
     clusters = km.fit_predict(df.iloc[:,1:])
     df["label"] = clusters
    
     fig = plt.figure(figsize=(20,5))
     ax = fig.add_subplot(111, projection='3d')
     ax.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1],km.cluster_centers_[:,2], color = 'red', s = 500, edgecolor= "red")
     ax.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1],km.cluster_centers_[:,2], color = 'red', s = 500, edgecolor = "red")
     ax.scatter(df.Age[df.label == 0], df["Annual Income"][df.label == 0], df["Spending Score"][df.label == 0], c='blue', s=60)
     ax.scatter(df.Age[df.label == 1], df["Annual Income"][df.label == 1], df["Spending Score"][df.label == 1], c='pink', s=60)
     ax.scatter(df.Age[df.label == 2], df["Annual Income"][df.label == 2], df["Spending Score"][df.label == 2], c='green', s=60)
     ax.scatter(df.Age[df.label == 3], df["Annual Income"][df.label == 3], df["Spending Score"][df.label == 3], c='orange', s=60)
     ax.scatter(df.Age[df.label == 4], df["Annual Income"][df.label == 4], df["Spending Score"][df.label == 4], c='purple', s=60)
     ax.view_init(30, 185)
     plt.title("Red datapoints are Cluster Centers")
     plt.xlabel("Age")
     plt.ylabel("Annual Income")
     ax.set_zlabel('Spending Score')
     st.pyplot(plt.show())


     st.subheader("Exporting the Clusters")

     
     vertical_stack = pd.concat([df.Age[df.label == 0], df["Annual Income"][df.label == 0], df["Spending Score"][df.label == 0] ], axis=1)
     csv1 = vertical_stack.to_csv().encode('utf-8')

     vertical_stack = pd.concat([df.Age[df.label == 1], df["Annual Income"][df.label == 1], df["Spending Score"][df.label == 1] ], axis=1)
     csv2 = vertical_stack.to_csv().encode('utf-8')

     vertical_stack = pd.concat([df.Age[df.label == 2], df["Annual Income"][df.label == 2], df["Spending Score"][df.label == 2] ], axis=1)
     csv3 = vertical_stack.to_csv().encode('utf-8')

     vertical_stack = pd.concat([df.Age[df.label == 3], df["Annual Income"][df.label == 3], df["Spending Score"][df.label == 3] ], axis=1)
     csv4 = vertical_stack.to_csv().encode('utf-8')

     vertical_stack = pd.concat([df.Age[df.label == 4], df["Annual Income"][df.label == 4], df["Spending Score"][df.label == 4] ], axis=1)
     csv5 = vertical_stack.to_csv().encode('utf-8')



     st.download_button( "Download Cluster 1", csv1,"./Cluster1.csv", "text/csv", key='download-csv')
     st.download_button( "Download Cluster 2", csv2,"./Cluster2.csv", "text/csv", key='download-csv')
     st.download_button( "Download Cluster 3", csv3,"./Cluster3.csv", "text/csv", key='download-csv')
     st.download_button( "Download Cluster 4", csv4,"./Cluster4.csv", "text/csv", key='download-csv')
     st.download_button( "Download Cluster 5", csv5,"./Cluster5.csv", "text/csv", key='download-csv')



     



         
     
            
      