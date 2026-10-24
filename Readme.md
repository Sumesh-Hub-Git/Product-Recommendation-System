# Product Recommendation System

A Python-based product recommendation project using Amazon Electronics user-rating data. The project explores popularity-based ranking and two personalized collaborative-filtering approaches: user-based collaborative filtering and matrix factorization with SVD/SVD++.

## Project approaches

1. **Popularity-based recommendation** — ranks products using rating frequency and average rating. This provides a non-personalized baseline.
2. **Preference-based recommendation** — uses user-based collaborative filtering and cosine similarity to find similar users, then predicts ratings using a similarity-weighted average.
3. **Preference-based SVD** — learns latent user and product factors from ratings, predicts ratings for unrated products, and ranks the predictions. SVD++ is also evaluated for comparison.

## Project files

- `popularity-based-product-recommendation-system.ipynb` — popularity-based recommendations and product-average rating baseline.
- `preference-based-product-recommendation-system.ipynb` — user-based collaborative filtering with cosine similarity.
- `preference-based-SVD-product-recommendation-system.ipynb` — SVD/SVD++ experiments, hyperparameter search, and SVD recommendations.
- `preference-based.md` — explanation of the user-based collaborative-filtering experiment and what was learned.
- `preference-based-SVD.md` — explanation of SVD/SVD++ experiments and what was learned.
- `README.md` — project overview, setup, results, and limitations.

## 1. Popularity-based recommendation

The popularity notebook ranks products using the number of ratings and their average ratings. It selects the 50 most frequently rated products, sorts them by average rating, and returns up to 10 recommendations.

This is a **non-personalized baseline**: the requested user ID labels the output but does not change the product ranking, so users generally receive the same recommendations.

The notebook also evaluates a product-average rating baseline. It predicts a test rating using the product's average rating from the training data, falling back to the training-set global average if the product was not present in training data.

## 2. Preference-based recommendation: user-based collaborative filtering

The preference notebook uses the first 3,000,000 rows of the dataset as its working subset, removes duplicates, and keeps users with at least 10 ratings. It splits the filtered interactions into training and test sets with a 70:30 split (`random_state=42`).

The `preference_based_recommendation(train_data, user_id, num_recommendations=5, k=20)` function:

1. Builds a user–product rating matrix from training data.
2. Calculates cosine similarity between the target user and other users.
3. Selects up to 20 other users with positive similarity.
4. Finds products the target user has not rated but at least one selected similar user has rated.
5. Predicts each candidate rating using a similarity-weighted average of neighbor ratings.
6. Ranks candidates by predicted rating and returns the requested number.

The output contains `productId`, `predicted_rating`, and `similar_users_count`.

### Evaluation

The notebook predicts eligible ratings from the held-out test set using similar users and reports the number of evaluated ratings, MSE, and RMSE. Rows are skipped if the user or product is absent from the training matrix, no similar users are available, or no selected neighbor rated the product. Similarities are recalculated for each eligible test interaction, so this evaluation can be slow.

See [preference-based.md](preference-based.md) for the detailed workflow, prediction formula, lessons learned, and limitations.

## 3. Preference-based SVD and SVD++

The SVD notebook uses the full filtered dataset rather than the smaller first-row subset used by the other two notebooks. It removes duplicate ratings and keeps users with at least 10 ratings, converts the ratings to Surprise's `Dataset` format, and creates a 70:30 train/test split.

The experiment trains SVD and SVD++, then evaluates both on the same test set. The recorded RMSE values were:

| Model | RMSE |
|---|---:|
| SVD | 1.0915 |
| SVD++ | 1.0977 |

Lower RMSE is better. SVD was marginally better in this run, but the difference is small and should not be treated as proof that SVD will always outperform SVD++.

The notebook also uses `GridSearchCV` to explore `n_factors`, `n_epochs`, `lr_all`, and `reg_all` with three-fold cross-validation. `GridSearchCV.fit()` requires a Surprise `Dataset`, not a `Trainset`. The final recommendation function predicts ratings for products the user has not rated, sorts them by estimated rating, and returns the top `n`.

**Tuning note:** the current notebook's final SVD instance uses fixed settings (`n_factors=50`, `n_epochs=20`) rather than the best parameters returned by the grid search. To apply the tuning result, initialize SVD using `**svd_search.best_params["rmse"]`. For a fair final evaluation, do not use the held-out test set during tuning.

See [preference-based-SVD.md](preference-based-SVD.md) for the prediction formula, SVD/SVD++ comparison, tuning details, and lessons learned.

## Dataset and preprocessing

The notebooks download the Amazon Product Reviews dataset through [`kagglehub`](https://github.com/Kaggle/kagglehub) and read the Electronics ratings CSV.

- Kaggle dataset: https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews
- Original Amazon Reviews data: http://jmcauley.ucsd.edu/data/amazon/

The data is represented by `userId`, `productId`, `rating`, and `timestamp`. The timestamp is removed for the recommendation experiments. Duplicate rows are removed, and users with fewer than 10 ratings are filtered out.

The popularity notebook uses the first 1,000,000 rows as a working subset; the user-based preference notebook uses the first 3,000,000 rows; the SVD notebook uses the full filtered dataset. Because the datasets differ in size, the reported experiments are not a perfectly controlled head-to-head comparison.

A user–product pivot matrix is also used for analysis, with missing entries filled with `0`. **A missing rating is not an actual rating of zero.** Evaluation should calculate errors only on known ratings in the held-out test data.

## Evaluation metrics

| Metric | What it measures | Preferred direction |
|---|---|---|
| **MSE** | Mean squared difference between actual and predicted ratings | Lower |
| **RMSE** | Square root of MSE, expressed in rating units | Lower |
| **Precision@K** | Fraction of the top-K recommended products that are relevant | Higher |
| **Recall@K** | Fraction of relevant products found in the top-K list | Higher |
| **NDCG@K** | Ranking quality that rewards relevant products appearing higher in the list | Higher |

MSE and RMSE measure rating-prediction accuracy; they do not directly measure top-N recommendation quality. Precision@K, Recall@K, and NDCG@K are useful additional ranking metrics. Their relevance threshold and candidate-item evaluation setup should be defined consistently across models.

## What we have learned so far

- **Popularity-based recommendations** provide a simple baseline but are not personalized.
- **User-based collaborative filtering** personalizes results by identifying similar users and weighting their ratings by similarity.
- **SVD** learns latent representations of users and products to predict missing ratings; those predictions can be ranked to produce recommendations.
- **SVD++** adds an implicit-feedback signal from a user's rated-product history, but it did not improve RMSE in the recorded experiment.
- **RMSE and top-N metrics answer different questions.** A model with lower rating error is not necessarily the model with the best top-10 ranking.
- **Evaluation design matters.** Compare methods on the same filtered data and test interactions where possible, tune only on training data, and reserve the test set for final evaluation.
- **Missing ratings must remain distinct from true ratings.** Zeros used to fill a pivot matrix are placeholders, not observed preferences.

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
- scikit-surprise (`surprise`)
- Jupyter Notebook

Depending on Python version and operating system, installing `scikit-surprise` may require compatible build tools or a supported Python environment.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sumesh-Hub-Git/Product-Recommendation-System.git
cd Product-Recommendation-System
```

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

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open the notebook you want to run and execute its cells in order. Dataset download requires internet access and may require any authentication or permissions requested by Kaggle.

## Limitations and future improvements

- Use the same dataset subset and held-out test interactions for a fair comparison between recommendation methods.
- Apply the best parameters found by `GridSearchCV` to the final SVD model, and keep test data separate from tuning.
- Add and compare Precision@10, Recall@10, and NDCG@10 for all personalized methods.
- Cache user similarities in the user-based model to reduce evaluation time.
- Consider sparse-matrix approaches to reduce memory usage on larger datasets.
- Test a hybrid recommender combining SVD predictions with content-based signals such as TF-IDF and cosine similarity.
- Consider a simple interface that accepts a user ID and displays recommended products.

## Notes

This is an educational recommendation-system project. Results depend on dataset preprocessing, data subset, train/test split, model settings, and evaluation procedure. The dataset is downloaded separately through `kagglehub` and is not necessarily stored in the repository.
