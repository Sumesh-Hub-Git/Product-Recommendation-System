# Preference-Based Recommendation Using SVD

## Overview

This experiment uses **Singular Value Decomposition (SVD) as a matrix-factorization recommender** from the `scikit-surprise` library. It learns latent numerical representations of users and products from known ratings, predicts ratings for user–product pairs, and recommends products the user has not rated.

The implementation is in `preference-based-SVD-product-recommendation-system.ipynb`.

## What we did

1. **Loaded the Amazon Electronics ratings data.** The notebook uses the ratings columns `userId`, `productId`, and `rating`; the timestamp is removed.
2. **Prepared the data.** Duplicate rows are removed and users with fewer than 10 ratings are filtered out. Unlike the popularity and user-based preference notebooks, this notebook uses the full filtered dataset rather than first restricting it to the first 1,000,000 or 3,000,000 rows.
3. **Converted ratings to Surprise format.** `Reader(rating_scale=(1, 5))` defines the rating range, and `Dataset.load_from_df(...)` creates a Surprise `Dataset`.
4. **Created a 70:30 train/test split.** Surprise's `train_test_split` creates a `Trainset` for fitting and a test list for evaluation.
5. **Trained SVD and SVD++.** Both models were fitted on the training set and evaluated against the same test set.
6. **Compared RMSE.** The notebook reports SVD RMSE of approximately **1.0915** and SVD++ RMSE of approximately **1.0977**. SVD was slightly better on this particular split, but the difference is small.
7. **Tried hyperparameter tuning.** `GridSearchCV` searches combinations of `n_factors`, `n_epochs`, `lr_all`, and `reg_all`, using three-fold cross-validation on a training-only Surprise `Dataset`.
8. **Built a recommendation function.** The final function identifies products in the filtered dataset that the target user has not rated, predicts a rating for each using `svd.predict(user_id, product_id)`, sorts predictions from highest to lowest, and returns the top `n` products.

## How SVD makes predictions

Surprise's SVD recommender is a biased matrix-factorization model. Its prediction has the form:

\[
\hat r_{ui} = \mu + b_u + b_i + q_i^T p_u
\]

Where:

- \(\hat r_{ui}\) is the predicted rating for user \(u\) and product \(i\).
- \(\mu\) is the global mean rating in the training data.
- \(b_u\) is the learned user bias.
- \(b_i\) is the learned product bias.
- \(p_u\) is the learned latent-factor vector for the user.
- \(q_i\) is the learned latent-factor vector for the product.
- \(q_i^T p_u\) measures how well the learned user and product factors match.

The latent factors are learned from rating examples during training. Their dimensions are numerical patterns; they are not automatically named concepts such as “price” or “quality.”

## SVD versus SVD++

The notebook also tests SVD++, which extends SVD by adding an implicit-feedback signal based on the products a user has rated. This extra signal can help, but it does not guarantee better results. In the recorded test split, SVD had the slightly lower RMSE.

| Model | Recorded RMSE | Interpretation |
|---|---:|---|
| SVD | 1.0915 | Slightly lower error in this experiment |
| SVD++ | 1.0977 | Slightly higher error in this experiment |

Lower RMSE is better for rating prediction. These values are specific to this run and split; they do not establish that SVD will always outperform SVD++.

## Hyperparameter tuning: what to know

The notebook's grid contains:

- `n_factors`: `[20, 50]`
- `n_epochs`: `[10, 20]`
- `lr_all`: `[0.002, 0.005]`
- `reg_all`: `[0.02, 0.05]`
- `cv=3`

`GridSearchCV.fit()` must receive a Surprise `Dataset`, not a `Trainset` or a test list. It makes its own cross-validation folds.

**Implementation note:** the notebook runs a separate pandas train/test split for tuning, then trains the final recommendation model with fixed settings (`n_factors=50`, `n_epochs=20`) on `data.build_full_trainset()`. The code shown does not apply the best parameters returned by `GridSearchCV` to that final model. To use the tuning result, initialize SVD with `**svd_search.best_params["rmse"]` and fit it on the intended training data. For final deployment, it is reasonable to retrain using all available ratings after model selection; for a final test score, keep the held-out test set untouched.

## Recommendation function

The notebook defines `recommend_products(user_id, n=10)`. It:

1. Finds products the user has already rated in `dataset_final`.
2. Finds all product IDs in `dataset_final`.
3. Removes already-rated products.
4. Predicts each remaining product's rating using the trained SVD model.
5. Sorts by estimated rating in descending order.
6. Returns the top `n` `(productId, predicted_rating)` pairs.

Example usage:

```python
recommendations = recommend_products(user_id, n=10)

for rank, (product_id, rating) in enumerate(recommendations, start=1):
    print(
        f"{rank}. Product ID: {product_id} | "
        f"Predicted Rating: {rating:.2f}"
    )
```

The function recommends only products represented in the filtered dataset. A user or product not known to the training model may receive a fallback prediction from Surprise, but the quality of that prediction may be limited.

## What we learned

- **Matrix factorization can predict missing ratings.** SVD learns latent user and product factors from the observed ratings rather than directly comparing every pair of users.
- **Predictions enable ranking.** After estimating ratings for unrated products, sorting those estimates produces a personalized top-N list.
- **More complex is not always better.** SVD++ uses extra interaction-history information, but it was not better by RMSE in this recorded experiment.
- **RMSE measures rating error, not recommendation ranking directly.** Precision@K, Recall@K, and NDCG@K should also be used to assess the top-ranked products.
- **Tuning must be evaluated carefully.** Use a training-only `Dataset` for cross-validation and keep a separate test set for final evaluation.
- **Runtime and scale matter.** SVD++ can be more computationally expensive than SVD, particularly on a large dataset.

## Main takeaway

SVD is a useful personalized recommendation model for this project: train it on historical ratings, predict ratings for unrated products, and rank those predictions. In the recorded comparison, SVD was chosen as the simpler model with marginally better RMSE, pending further tuning and top-N evaluation.
