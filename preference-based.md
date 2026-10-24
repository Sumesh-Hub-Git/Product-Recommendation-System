# Preference-Based Product Recommendation

## Overview

This experiment builds a personalized recommender using **user-based collaborative filtering**. Instead of recommending the same popular products to everyone, it looks for users whose rating patterns resemble those of the target user and uses their ratings to estimate which unseen products the target user may like.

The implementation is in `preference-based-product-recommendation-system.ipynb`.

## What we did

1. **Loaded the Amazon Electronics ratings data.** The data contains `userId`, `productId`, `rating`, and `timestamp`. The timestamp was removed because it was not used by this experiment.
2. **Reduced the working data.** The notebook takes the first 3,000,000 rows, removes duplicate records, and keeps users who have at least 10 ratings.
3. **Split the data.** The filtered interactions are divided into training and test sets using a 70:30 split with `random_state=42`.
4. **Built a user–product matrix.** Rows represent users, columns represent products, and cells contain ratings. Missing ratings are filled with zero for matrix operations.
5. **Found similar users.** Cosine similarity compares the target user's rating vector with other users' vectors. The target user is removed from the neighbor list, and up to the 20 most similar users with positive similarity are selected.
6. **Found candidate products.** A candidate must be unrated by the target user but rated by at least one selected similar user.
7. **Predicted ratings.** For each candidate product, ratings from similar users are combined using similarity-weighted averaging. Users with greater similarity have more influence on the prediction.
8. **Ranked recommendations.** Candidate products are sorted by predicted rating, and the requested number of highest-ranked products is returned.

## How the prediction works

For a candidate product \(i\), the predicted rating is a similarity-weighted average:

\[
\hat r_{u,i} =
\frac{\sum_{v \in N_i(u)} s(u,v)\,r_{v,i}}
{\sum_{v \in N_i(u)} s(u,v)}
\]

Where:

- \(u\) is the target user.
- \(v\) is a similar user.
- \(s(u,v)\) is the cosine similarity between the two users.
- \(r_{v,i}\) is the rating that similar user \(v\) gave product \(i\).
- \(N_i(u)\) contains selected similar users who rated product \(i\).

Only neighbors who rated the candidate product contribute to its prediction.

## Recommendation output

The function `preference_based_recommendation(...)` returns a DataFrame with:

| Column | Meaning |
|---|---|
| `productId` | Recommended product identifier |
| `predicted_rating` | Similarity-weighted predicted rating |
| `similar_users_count` | Number of selected similar users who rated that product |

The notebook demonstrates recommendations for three sampled users. Because users have different rating histories and different nearest neighbors, their recommendations can differ.

## Evaluation performed

The notebook evaluates rating predictions on eligible interactions from the held-out test set:

- Similar users are identified using the training matrix.
- Ratings from neighbors who rated the test product are combined into a prediction.
- Rows are skipped when the user or product is absent from the training matrix, no positive-similarity neighbors are available, or no usable neighbor ratings exist.
- MSE and RMSE are calculated from the actual and predicted ratings that could be evaluated.

The notebook prints the number of evaluated test ratings, MSE, and RMSE when eligible examples exist. Exact evaluation values depend on executing the notebook; this write-up does not assume a result that was not provided.

## What we learned

- **Personalization comes from user similarity.** The ranking depends on the target user's rating history, unlike popularity-based recommendations.
- **Neighbor quality matters.** A similar user's rating should influence a prediction more when their rating pattern is more similar to the target user's.
- **Recommendations need evidence.** This implementation recommends only products rated by at least one selected neighbor; it does not score every product in the catalogue.
- **Sparse data is a challenge.** Many users have rated different products, so there may be little overlap from which to calculate meaningful similarities.
- **Cold-start cases are limited.** A user absent from the training matrix raises an error, and products absent from the matrix cannot be recommended by this implementation.
- **Evaluation is not the same as top-N ranking evaluation.** MSE/RMSE assess rating prediction error. Precision@K, Recall@K, or NDCG@K would be useful additional measures for the quality of the final ranked list.

## Limitations and possible improvements

- The user–product matrix can consume a lot of memory as the number of users and products grows.
- Cosine similarities are recalculated for individual test interactions in the evaluation loop, which can make evaluation slow.
- Missing ratings are represented by zero in the pivot matrix. The code treats zero as “not rated”; it must not be interpreted as an actual user rating.
- The model only considers up to 20 positively similar users. The best value of `k` could be tested on validation data.
- For a fair comparison with other methods, use the same data subset and held-out test interactions.
- Add top-N metrics such as Precision@10, Recall@10, and NDCG@10.

## Main takeaway

This approach is an interpretable baseline for personalized recommendations: find users with similar rating patterns, use their ratings to predict candidate products, and rank the products by predicted rating. Its success depends on overlap between users' rating histories.
