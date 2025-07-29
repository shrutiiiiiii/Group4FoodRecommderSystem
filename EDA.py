# =============================================
# 0. SETUP & DATA LOADING
# =============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load data
path = "irish_restaurants_meals_final.csv"
df = pd.read_csv(path)

# Preprocessing - more comprehensive NaN handling
df.dropna(subset=['Rating', 'Price', 'Latitude', 'Longitude'], inplace=True)
df['Meal_Category'] = df['Meal'].apply(lambda x: x.split()[0])  # Extract first word as category
df['Price_Bin'] = pd.cut(df['Price'], bins=5, labels=['Cheap', 'Low', 'Medium', 'High', 'Premium'])

# Data validation
print(f"\nData shape after cleaning: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Unique restaurants: {df['Restaurant'].nunique()}")
print(f"Unique meals: {df['Meal'].nunique()}\n")

# =============================================
# 1. COSINE SIMILARITY ANALYSIS (Content-Based)
# =============================================
print("\n=== 1. COSINE SIMILARITY ANALYSIS ===")

# Improved Feature Engineering
df['Price_Norm'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
df['Rating_Norm'] = (df['Rating'] - df['Rating'].mean()) / df['Rating'].std()
features = df[['Price_Norm', 'Rating_Norm', 'Latitude', 'Longitude']]

# Compute Similarity Matrix
similarity_matrix = cosine_similarity(features)

# VISUALIZATION 1: Heatmap of Similarity Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix[:20, :20], cmap='viridis', annot=True, fmt=".2f")
plt.title("Cosine Similarity Between Top 20 Meals")
plt.show()

# VISUALIZATION 2: Similarity Distribution
plt.figure(figsize=(10, 6))
sns.histplot(similarity_matrix.flatten(), bins=50, kde=True)
plt.title("Distribution of Cosine Similarity Scores")
plt.xlabel("Similarity Score")
plt.show()

# VISUALIZATION 3: Top Similar Pairs
top_pairs = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i,j] > 0.9:  # High similarity threshold
            top_pairs.append((df.iloc[i]['Meal'], df.iloc[j]['Meal'], similarity_matrix[i,j]))
            
if top_pairs:
    top_pairs_df = pd.DataFrame(top_pairs, columns=['Meal1', 'Meal2', 'Similarity']).sort_values('Similarity', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Similarity', y='Meal1', data=top_pairs_df.head(10))
    plt.title("Top 10 Most Similar Meal Pairs")
    plt.show()
else:
    print("No highly similar meal pairs found with threshold > 0.9")

# =============================================
# 2. ASSOCIATION RULE MINING (Market Basket)
# =============================================
print("\n=== 2. ASSOCIATION RULE MINING ===")

# Prepare Transaction Data with more categories
df['Meal_Category'] = df['Meal'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'Other')
pivot = pd.crosstab(df['Restaurant'], df['Meal_Category']).applymap(lambda x: 1 if x > 0 else 0)

# Only run if we have enough data
if len(pivot) > 5 and pivot.sum().sum() > 0:
    # Apriori Algorithm with adjusted parameters
    frequent_itemsets = apriori(pivot, min_support=0.05, use_colnames=True, max_len=3)
    
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        if len(rules) > 0:
            # VISUALIZATION 1: Support vs Confidence
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='support', y='confidence', size='lift', 
                          data=rules, hue='lift', palette='viridis')
            plt.title("Association Rules: Support vs Confidence")
            plt.show()

            # VISUALIZATION 2: Top Rules by Lift
            plt.figure(figsize=(12, 6))
            top_rules = rules.sort_values('lift', ascending=False).head(10)
            top_rules['rule'] = top_rules['antecedents'].astype(str) + " -> " + top_rules['consequents'].astype(str)
            sns.barplot(x='lift', y='rule', data=top_rules)
            plt.title("Top 10 Association Rules by Lift Score")
            plt.tight_layout()
            plt.show()

            # VISUALIZATION 3: Parallel Categories Plot
            try:
                import plotly.express as px
                fig = px.parallel_categories(rules.head(10), 
                                          dimensions=['antecedents', 'consequents'],
                                          color='lift', 
                                          title="Association Rules Flow")
                fig.show()
            except:
                print("Plotly not available for parallel categories plot")
        else:
            print("No association rules found with current parameters")
    else:
        print("No frequent itemsets found with current parameters")
else:
    print("Not enough data for meaningful association rule mining")

# =============================================
# 3. K-MEANS CLUSTERING (Segmentation)
# =============================================
print("\n=== 3. K-MEANS CLUSTERING ===")

# Prepare Data
X = df[['Price', 'Rating', 'Latitude', 'Longitude']].drop_duplicates()

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means (K=4)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)
X['Cluster'] = clusters

# VISUALIZATION 1: 2D Geographic Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', data=X, palette='viridis', s=100)
plt.title("Restaurant Clusters in Dublin (Geographic)")
plt.show()

# VISUALIZATION 2: PCA Reduced Dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[['Price', 'Rating', 'Latitude', 'Longitude']])
X['PCA1'] = X_pca[:, 0]
X['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=X, palette='viridis', s=100)
plt.title("Restaurant Clusters (PCA Reduced Dimensions)")
plt.show()

# VISUALIZATION 3: Cluster Characteristics
cluster_stats = X.groupby('Cluster')[['Price', 'Rating']].mean()
cluster_stats.plot(kind='bar', figsize=(10, 6))
plt.title("Average Price and Rating by Cluster")
plt.ylabel("Value")
plt.show()

# =============================================
# 4. FACTOR ANALYSIS (Latent Dimensions)
# =============================================
print("\n=== 4. FACTOR ANALYSIS ===")

# Factor Analysis
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(X[['Price', 'Rating', 'Latitude', 'Longitude']])

# Get Loadings
loadings = pd.DataFrame(fa.loadings_, 
                       columns=['Factor_1', 'Factor_2', 'Factor_3'], 
                       index=['Price', 'Rating', 'Latitude', 'Longitude'])

# VISUALIZATION 1: Factor Loadings Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, cmap='coolwarm', annot=True, fmt=".2f", center=0)
plt.title("Factor Loadings Matrix")
plt.show()

# VISUALIZATION 2: Scree Plot (Fixed)
ev, v = fa.get_eigenvalues()
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ev)+1), ev, marker='o')
plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.axhline(y=1, color='r', linestyle='--')
plt.show()

# VISUALIZATION 3: Factor Scores
factor_scores = fa.transform(X[['Price', 'Rating', 'Latitude', 'Longitude']])
X[['Factor1', 'Factor2', 'Factor3']] = factor_scores

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['Factor1'], X['Factor2'], X['Factor3'], c=X['Cluster'], cmap='viridis')
ax.set_xlabel('Factor 1 (Price)')
ax.set_ylabel('Factor 2 (Rating)')
ax.set_zlabel('Factor 3 (Location)')
plt.title("3D Factor Space")
plt.show()

# =============================================
# 5. K-NEAREST NEIGHBORS (Spatial Analysis)
# =============================================
print("\n=== 5. K-NEAREST NEIGHBORS ===")

# KNN Model
coords = df[['Latitude', 'Longitude']].drop_duplicates().values
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(coords)

# Example Query (Dublin City Center)
test_point = np.array([[53.3498, -6.2603]])
distances, indices = knn.kneighbors(test_point)

# VISUALIZATION 1: Nearest Restaurants Map
plt.figure(figsize=(12, 8))
sns.scatterplot(x=coords[:, 1], y=coords[:, 0], label='All Restaurants')
sns.scatterplot(x=coords[indices[0], 1], y=coords[indices[0], 0], color='red', s=200, label='Nearest 5')
plt.scatter(test_point[0][1], test_point[0][0], color='black', marker='*', s=300, label='Query Point')
plt.title("KNN: 5 Nearest Restaurants to Dublin City Center")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# VISUALIZATION 2: Distance Distribution
plt.figure(figsize=(10, 6))
sns.histplot(distances.flatten(), bins=20, kde=True)
plt.title("Distribution of KNN Distances")
plt.xlabel("Distance (km)")
plt.show()

# VISUALIZATION 3: Price-Rating of Nearest Neighbors
nearest_data = df.iloc[indices[0]]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Rating', data=nearest_data, hue='Meal_Category', s=100)
plt.title("Price vs Rating of Nearest Restaurants")
plt.show()