# Product Recommendation System

A Python-based product recommendation project using Amazon Electronics
user-rating data. The notebook explores the dataset, analyzes rating
patterns, builds a popularity-based recommendation baseline, and
evaluates rating predictions with Mean Squared Error (MSE) and Root Mean
Squared Error (RMSE).

## Project Overview

The project starts with a **non-personalized, popularity-based
recommender**. It identifies products that have received many ratings
and ranks the most popular products using their average ratings.

A planned/alternative approach is **preference-similarity-based
recommendation**, which aims to personalize suggestions using patterns
in users' ratings. This can be implemented with collaborative filtering
methods such as user-based similarity or matrix-factorization models
such as SVD/SVD++.

> **Current status:** The notebook currently implements the
> popularity-based recommendation function and a product-average rating
> baseline for MSE/RMSE evaluation. Preference-similarity methods are an
> extension unless you have added and run them separately.

## Current Recommendation Method

The notebook's `recommend(train_data, user_id)` function:

1.  Counts how many ratings each product received in the training data.
2.  Calculates each product's average rating.
3.  Selects the 50 most frequently rated products.
4.  Sorts those products by average rating.
5.  Returns recommendations associated with the requested user ID.

Because this method is popularity-based, users generally receive the
same product ranking. The user ID labels the output but does not change
the ranking based on that user's personal preferences.

### Limitation

A highly rated product is not necessarily a good match for every user.
The current method does not learn an individual's taste from their
rating history.

## Planned Extension: Preference-Similarity-Based Recommendation

The next method can personalize recommendations by learning from users'
rating patterns.

A typical workflow is:

1.  Create a user--product rating matrix.
2.  Compare users' preferences or learn latent user/product factors.
3.  Predict ratings for products the target user has not rated.
4.  Rank candidate products by predicted rating.
5.  Recommend the highest-ranked products, excluding products the user
    has already rated.

Possible approaches include:

-   **User-based collaborative filtering:** recommends products liked by
    users with similar rating patterns.
-   **SVD:** learns latent factors for users and products to predict
    ratings.
-   **SVD++:** extends SVD by also using implicit feedback from a user's
    item interactions represented in the training ratings data.

These approaches should be evaluated separately from the
popularity-based baseline using the same held-out test set.

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

The notebook uses Python and common data-science libraries, including:

-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   SciPy
-   joblib
-   kagglehub
-   scikit-surprise (`surprise`)

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/Sumesh-Hub-Git/Product-Recommendation-System.git
cd <your-repository-folder>
```

Replace the placeholders with your repository URL and folder name.

### 2. Create and activate a virtual environment (recommended)

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib kagglehub scikit-surprise jupyter
```

Depending on your Python version and operating system, installing
`scikit-surprise` may require compatible build tools or a supported
Python environment.

### 4. Launch Jupyter

``` bash
jupyter notebook
```

Open `product-recommendation-system.ipynb` and run the notebook cells in
order.

## Dataset Access

The notebook uses
`kagglehub.dataset_download("saurav9786/amazon-product-reviews")`. You
may need internet access and any authentication or permissions required
by Kaggle. Check the downloaded directory and update the CSV path if the
file name or directory structure differs from the notebook's expected
path.

## Project Structure

``` text
.
├── product-recommendation-system.ipynb
└── README.md
```

The dataset is downloaded separately by `kagglehub`; it is not
necessarily stored inside the repository.

## Future Improvements

-   Implement and compare user-based collaborative filtering, SVD, and
    SVD++.
-   Add MAE alongside MSE and RMSE.
-   Evaluate top-N recommendations with Precision@K, Recall@K, and
    NDCG@K.
-   Ensure every model uses the same train/test split for fair
    comparison.
-   Avoid recommending products the target user has already rated.
-   Consider sparse-matrix techniques for large user--product datasets.
-   Add a simple interface for entering a user ID and viewing
    recommendations.

## Notes

This is an educational recommendation-system project. Results depend on
the selected subset, preprocessing choices, data split, and model
configuration.
