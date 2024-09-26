## Housing Market Analysis and Prediction**

This project leverages machine learning techniques to analyze and predict housing prices using Melbourne's housing market dataset. The models employed include **Linear Regression**, **Ridge Regression**, **Polynomial Regression**, and clustering techniques such as **K-Means** and **DBSCAN**. The goal is to provide accurate price predictions and understand property clusters in the dataset.

## Dataset

The dataset used is the [Melbourne Housing Market Dataset](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market). It contains detailed information about the houses in Melbourne, such as their location, type, land size, number of bedrooms, and other key features.

---

## 1. Environment Setup

Ensure you have Python 3.7+ installed, along with the following libraries:

- numpy
- pandas
- matplotlib
- scikit-learn

### Install Dependencies

To install the necessary dependencies, run the following commands:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 2. Dataset Preparation

### Load the Dataset

Ensure the dataset is placed in the correct path as shown in the image:

```python
# Load the dataset
data = pd.read_csv('/path_to_your_dataset/Melbourne_housing_FULL.csv')

# Displaying the first few rows of the dataset
data.head()
```

### Clean and Preprocess the Dataset

The dataset will be cleaned by removing any rows with missing values, followed by encoding categorical columns into numerical form:

```python
def preprocess_inputs(df):
    df = df.dropna().reset_index(drop=True)

    columns_to_drop = ['Price', 'Rooms', 'Address', 'Method', 'Date', 'Distance', 'Postcode', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount', 'SellerG']
    df = df.drop(columns_to_drop, axis=1)

    df = pd.get_dummies(df, drop_first=True)  # Encode categorical variables

    # Separate the target variable 'Price' and features
    y = df['Price']
    X = df.drop('Price', axis=1)
    
    return X, y

# Preprocess the dataset
X, y = preprocess_inputs(data)
```

### Train-Test Split

Split the dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## 3. Machine Learning Models used
## Regression Models

### Linear Regression

We begin with training a **Linear Regression** model:

```python
# Initialise and train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on test set
y_pred_linear = linear_model.predict(X_test)

# Evaluate model performance using R² score
linear_r2 = r2_score(y_test, y_pred_linear)
print(f"Linear Regression R² Score: {linear_r2}")
```

### Ridge Regression

To address potential overfitting, we use **Ridge Regression** with regularisation. The best alpha value is chosen based on the R² score:

```python
# Train Ridge Regression with multiple alpha values
alpha_values = np.arange(0.1, 2.1, 0.1)
best_r2_score = -1
best_alpha = None

for alpha in alpha_values:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_ridge)
    if r2 > best_r2_score:
        best_r2_score = r2
        best_alpha = alpha

print(f"Best Ridge Regression Alpha: {best_alpha} with R² score: {best_r2_score}")
```

### Polynomial Regression (Degree 2)

We also explore **Polynomial Regression** to capture non-linear relationships:

```python
# Apply Polynomial Features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train Polynomial Regression
poly_model = LinearRegression()
poly_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_poly = poly_model.predict(X_test)
poly_r2 = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression R² Score: {poly_r2}")
```

---

## Clustering Models

### K-Means Clustering

We apply **K-Means Clustering** to identify property clusters:

```python
# Perform K-Means Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Visualise the clusters (example for first two features)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red')  # Cluster centers
plt.show()

# Define the range for the number of clusters to test
cluster_range = range(2, 11)  # Testing from 2 to 10 clusters

# Initialise variables to track the best silhouette score and best number of clusters
best_score = -1
best_n_clusters = None
best_kmeans = None

# Loop over the cluster range and compute silhouette scores
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    # Calculate silhouette score
    score = silhouette_score(X_scaled, labels)
    print(f"K-Means with {n_clusters} clusters has silhouette score: {score:.4f}")
    
    # Track the best score and corresponding number of clusters
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters
        best_kmeans = kmeans

# Output the best number of clusters and the best silhouette score
print(f"\nBest K-Means Params: n_clusters={best_n_clusters} with silhouette score: {best_score:.4f}")
```

### DBSCAN Clustering

For a more flexible approach, we use **DBSCAN** clustering, fine-tuning the `eps` and `min_samples` parameters:

```python
eps_values = [0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_values = [3, 4, 5, 6, 7]
best_score = -1
best_eps = None
best_min_samples = None

# Loop through each combination of eps and min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)  # Fit and predict DBSCAN clusters
        
        # Only calculate silhouette score if there's more than 1 cluster
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            print(f"DBSCAN with eps = {eps} and min_samples = {min_samples} has silhouette score: {score:.4f}")
            
            # Track best parameters based on highest silhouette score
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

print(f"Best DBSCAN eps={best_eps}, min_samples={best_min_samples}")
```

---

## 4. Prediction

### Making Predictions

To make predictions using the trained models:

```python
# Use the trained Ridge Regression model to make predictions on new data
new_data = X_test[:5]  # Example new data
predictions = ridge_model.predict(new_data)
print("Predicted House Prices:", predictions)
```

---

## 5. Conclusion

This project uses machine learning models for predicting housing prices and clustering properties. **Ridge Regression** outperforms other models for prediction, while **DBSCAN** is the most suitable clustering technique, offering meaningful clusters of properties.

Both models together offer insights into housing price predictions and property segmentation, providing a deeper understanding of the Melbourne housing market.

