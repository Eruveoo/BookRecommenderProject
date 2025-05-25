import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.empty:
        return pd.Series([], dtype=float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_book_recommendations(liked_isbns, user_item_matrix_df, book_metadata_dict,
                                  top_n_similar_users=30, num_recs=5):
    """
    Generates book recommendations based on liked ISBNs.

    Args:
        liked_isbns (list): A list of ISBNs for books the user likes.
        user_item_matrix_df (pd.DataFrame): The user-item rating matrix.
        book_metadata_dict (dict): A dictionary mapping ISBNs to book metadata.
        top_n_similar_users (int): The number of similar users to consider.
        num_recs (int): The number of recommendations to return.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'title', 'author', 'isbn', and 'score' for a recommended book.
              Returns an empty list if no recommendations can be made.
    """
    if not liked_isbns:
        print("[DEBUG] No liked ISBNs provided.")
        return []

    new_user_vector = pd.Series(0, index=user_item_matrix_df.columns, dtype=float)

    found_liked_books_in_matrix = False
    for isbn in liked_isbns:
        if isbn in new_user_vector.index:
            new_user_vector[isbn] = 1.0
            found_liked_books_in_matrix = True
        else:
            print(f"[WARN] ISBN {isbn} (liked by user) not found in user-item matrix columns.")

    if not found_liked_books_in_matrix:
        print("[DEBUG] None of the liked ISBNs were found in the matrix. Cannot generate recommendations.")
        return []

    user_matrix_for_similarity = user_item_matrix_df.fillna(0).values
    new_user_np = new_user_vector.values.reshape(1, -1)

    similarities = cosine_similarity(new_user_np, user_matrix_for_similarity)[0]
    similarity_series = pd.Series(similarities, index=user_item_matrix_df.index)

    positive_similarity_users = similarity_series[similarity_series > 0]
    top_similar_users_with_sim = positive_similarity_users.sort_values(ascending=False).head(top_n_similar_users)

    if top_similar_users_with_sim.empty:
        print("[DEBUG] No similar users found with positive similarity.")
        return []


    similar_users_ratings = user_item_matrix_df.loc[top_similar_users_with_sim.index]
    user_mean_ratings = user_item_matrix_df.mean(axis=1)

    potential_recommendation_books_unliked = user_item_matrix_df.columns.difference(liked_isbns)
    similar_users_ratings_for_unliked = similar_users_ratings[potential_recommendation_books_unliked]

    books_rated_by_any_similar_user_unliked = similar_users_ratings_for_unliked.columns[
        similar_users_ratings_for_unliked.notna().any()
    ]
    similar_users_ratings_filtered = similar_users_ratings_for_unliked[books_rated_by_any_similar_user_unliked]


    if similar_users_ratings_filtered.empty:
        print("[DEBUG] Similar users have not rated any new books relevant for recommendation.")
        return []

    predicted_book_scores = pd.Series(0.0, index=similar_users_ratings_filtered.columns)

    sum_abs_similarities = top_similar_users_with_sim.abs().sum()

    if sum_abs_similarities == 0:
        print("[DEBUG] Sum of absolute similarities is zero. Cannot compute predictions.")
        return []

    for book_isbn in predicted_book_scores.index:
        weighted_deviation_sum = 0.0
        for user_id, similarity_score in top_similar_users_with_sim.items():
            user_rating = similar_users_ratings_filtered.loc[user_id, book_isbn]
            if pd.notna(user_rating):
                user_mean = user_mean_ratings.loc[user_id]
                deviation = user_rating - user_mean
                weighted_deviation_sum += similarity_score * deviation

        predicted_book_scores[book_isbn] = weighted_deviation_sum / sum_abs_similarities

    book_scores = predicted_book_scores[predicted_book_scores > 0]
    if book_scores.empty:
        print("[DEBUG] All potential recommendations had a predicted score of 0 or less.")
        return []

    softmax_scores = softmax(book_scores)

    top_recommendations_series = softmax_scores.sort_values(ascending=False).head(num_recs)

    recommendations_list = []
    for isbn, score in top_recommendations_series.items():
        book_details = book_metadata_dict.get(isbn, {})
        title = book_details.get("Book-Title", "Unknown Title")
        author = book_details.get("Book-Author", "Unknown Author")
        recommendations_list.append({
            "title": title,
            "author": author,
            "isbn": isbn,
            "score": score
        })

    print(f"Generated {len(recommendations_list)} recommendations.")
    return recommendations_list