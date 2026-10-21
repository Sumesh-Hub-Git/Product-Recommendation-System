# Product Recommendation System

A Python-based product recommendation project using Amazon Electronics user-rating data. The project contains two Jupyter notebooks that explore the dataset and implement two recommendation approaches:

1. **Popularity-Based Recommendation** — ranks products using rating frequency and average ratings.
2. **Preference-Based Recommendation** — uses user-based collaborative filtering with cosine similarity to recommend products based on the ratings of similar users.

Both notebooks evaluate rating-prediction performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Project Files

- `popularity-based-product-recommendation-system.ipynb` — popularity-based recommendations and a product-average rating prediction baseline.
- `preference-based-product-recommendation-system(3).ipynb` — personalized recommendations using user-based collaborative filtering and cosine similarity.
- `README.md` — project description and setup instructions.

## 1. Popularity-Based Recommendation

The popularity notebook implements a `recommend(train_data, user_id)` function. It:

1. Counts the number of ratings recorded for each product in the training data.
2. Calculates each product's average rating.
3. Selects the 50 most frequently rated products.
4. Sorts those products by average rating and returns up to 10 recommendations.
5. Adds the requested user ID to the output.

This is a **non-personalized baseline**: the requested user ID labels the results, but it does not affect the product ranking. Users therefore generally receive the same product recommendations.

The notebook also evaluates a product-average rating baseline. It predicts a test rating using the product's average rating in the training data, falling back to the training-set global average if the product was not present in training data.

## 2. Preference-Based Recommendation Using User-Based Collaborative Filtering

The preference notebook implements `preference_based_recommendation(train_data, user_id, num_recommendations=5, k=20)`. It personalizes recommendations using user rating patterns:

1. Builds a user–product rating matrix from **training data only**.
2. Calculates cosine similarity between the target user and the users in the matrix.
3. Selects up to 20 other users with the highest positive similarity scores.
4. Finds products that the target user has not rated but at least one selected similar user has rated.
5. Predicts a rating for each candidate product using a similarity-weighted average of the selected users' ratings for that product.
6. Sorts candidates by predicted rating and returns the requested number of recommendations.

The recommendation output contains:

- `productId` — the recommended product identifier.
- `predicted_rating` — the model's similarity-weighted predicted rating.
- `similar_users_count` — the number of selected similar users who rated that product.

The notebook demonstrates recommendations for three sampled users. Since the candidate products and rating weights depend on the target user's nearest neighbors, different users can receive different recommendations.

### Preference Model Evaluation

The notebook evaluates predictions on held-out rows from `test_data`. For each eligible test interaction, it finds the target user's most similar users from the training matrix and predicts the rating from neighbors who rated that product. Rows are skipped when the user or product is not present in the training matrix, when there are no similar users, or when no usable neighbor ratings are available. The notebook reports the number of evaluated ratings and the resulting MSE and RMSE.

This evaluation may take time on a large dataset because similarities are recalculated for individual test rows.

## Dataset and Preprocessing

The notebook downloads the Amazon Product Reviews dataset using
[`kagglehub`](https://github.com/Kaggle/kagglehub) and reads the
Electronics ratings CSV.

Link : https://www.kaggle.com/saurav9786/amazon-product-reviews?fbclid=IwAR26xfJJK1c5zEeXh8yK9tTgDZrXSkdZIIfjGOhnegCobgskrUEYFcky_cw

The dataset here is taken from the below website.

Source - Amazon Reviews data (http://jmcauley.ucsd.edu/data/amazon/) The repository has several datasets. For this case study, we are using the Electronics dataset.

The notebook then:

-   Assigns the columns `userId`, `productId`, `rating`, and
    `timestamp`.
-   Inspects data types, missing values, and dataset dimensions.
-   Removes the timestamp column for the recommendation analysis.
-   Uses the first 1,000,000 rows as a smaller working subset.
-   Removes duplicate records.
-   Keeps users with at least 10 ratings.
-   Examines rating distributions and product popularity.
-   Creates a user--product pivot matrix, filling missing entries with
    `0` for matrix exploration.
-   Splits the filtered interactions into training and test sets using a
    70:30 split (`random_state=42`).

**Important:** A missing rating is not the same as a real rating of
zero. When evaluating rating-prediction models, calculate errors only
for known ratings in the held-out test set, not for the zeros used to
represent missing entries in the pivot matrix.

## Evaluation

The notebook evaluates a product-average rating baseline:

-   For each product, calculate its mean rating using the training data.
-   For a test interaction, predict that product's training-set mean.
-   If a product does not appear in the training data, use the global
    mean rating from the training data as a fallback.
-   Compare predictions with the actual test ratings.

### Metrics

  -----------------------------------------------------------------------
  Metric                  Description             Preferred direction
  ----------------------- ----------------------- -----------------------
  **MSE**                 Mean of squared         Lower is better
                          differences between     
                          actual and predicted    
                          ratings                 

  **RMSE**                Square root of MSE;     Lower is better
                          error is expressed in   
                          rating units            

  -----------------------------------------------------------------------

MSE and RMSE measure rating-prediction accuracy. They do not directly
measure whether the top recommended products are relevant. For top-N
recommendation quality, consider metrics such as **Precision@K**,
**Recall@K**, and **NDCG@K**.

## Requirements

The notebooks use Python and the following main libraries:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- SciPy
- joblib
- kagglehub
- scikit-surprise (`surprise`), imported in the notebooks
- Jupyter Notebook

Some imports, including Surprise/SVD-related classes, are present even though the two described recommendation functions do not implement an SVD recommender.

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/Sumesh-Hub-Git/Product-Recommendation-System.git
cd Sumesh-Hub-Git/Product-Recommendation-System
```

Replace the placeholders with your repository URL and folder name.

### 2. Create and activate a virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib kagglehub scikit-surprise jupyter
```

Depending on your Python version and operating system, installing `scikit-surprise` may require compatible build tools or a supported Python environment.

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open either notebook and run its cells in order. The dataset download requires internet access and may require any authentication or permissions requested by Kaggle.

## Limitations and Future Improvements

- The popularity recommender is not personalized; the user ID does not change its ranking.
- The preference-based method uses cosine similarity and a weighted average of neighbors' ratings; its quality depends on the amount of overlap in users' rating histories.
- The preference notebook's evaluation can be slow because it recomputes cosine similarities for each eligible test row; caching neighbors would improve performance.
- The notebooks currently use different initial subset sizes. Use the same subset and test interactions for a fair direct comparison.
- Consider evaluating top-N relevance with Precision@K, Recall@K, and NDCG@K.
- Consider sparse-matrix techniques to reduce memory usage for larger datasets.
- An SVD/SVD++ implementation and evaluation would be a separate extension; importing those libraries alone does not mean those models are implemented.
- A simple interface could allow users to enter a user ID and view recommendations.

## Notes

This is an educational recommendation-system project. Results depend on the selected dataset subset, preprocessing, train/test split, and model configuration. The dataset is downloaded separately through `kagglehub` and is not necessarily stored inside the repository.
