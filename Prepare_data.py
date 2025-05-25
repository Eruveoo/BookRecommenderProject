import pandas as pd
import re

MIN_USER_RATINGS = 100
MIN_BOOK_RATINGS = 5
TOP_N_BOOKS = 20000

books = pd.read_csv("books.csv", dtype=str)
ratings = pd.read_csv("ratings.csv")

print(f"Loaded books: {books.shape}, ratings: {ratings.shape}")
print(f"Books:\n{books.head(3)}")
print(f"Ratings:\n{ratings.head(3)}")

# Keep only explicit ratings
ratings_before = ratings.shape[0]
ratings = ratings[ratings["Book-Rating"] > 0]
print(f"Removed {ratings_before - ratings.shape[0]} implicit (0) ratings. Remaining: {ratings.shape[0]}")


user_counts = ratings["User-ID"].value_counts()
active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
print(f"Found {len(active_users)} active users with ≥ {MIN_USER_RATINGS} ratings.")
ratings = ratings[ratings["User-ID"].isin(active_users)]

# Deduplicate
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[\.\,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text.strip()

def normalize_author(text):
    text = str(text).lower()
    text = re.sub(r'\.', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text.strip()

print("Normalizing book titles and authors...")
books["NormalizedTitle"] = books["Book-Title"].apply(normalize_text)
books["NormalizedAuthor"] = books["Book-Author"].apply(normalize_author)
books["DuplicateKey"] = books["NormalizedTitle"] + "|" + books["NormalizedAuthor"]

isbn_rating_counts = ratings["ISBN"].value_counts().to_dict()
books["RatingCount"] = books["ISBN"].map(isbn_rating_counts).fillna(0).astype(int)

print("Deduplicating books...")
deduplicated_books = books.sort_values("RatingCount", ascending=False).drop_duplicates("DuplicateKey")
print(f"Removed {books.shape[0] - deduplicated_books.shape[0]} book duplicates")

duplicate_map = {}
for key, group in books.groupby("DuplicateKey"):
    main_isbn = group.sort_values("RatingCount", ascending=False).iloc[0]["ISBN"]
    for isbn in group["ISBN"]:
        duplicate_map[isbn] = main_isbn

# Remapping in ratings
print(f"Remapping {len(duplicate_map)} ISBNs in ratings...")
ratings["ISBN"] = ratings["ISBN"].map(duplicate_map).fillna(ratings["ISBN"])

# Filter popular groups
book_counts = ratings["ISBN"].value_counts()
popular_books = book_counts[book_counts >= MIN_BOOK_RATINGS].sort_values(ascending=False)
print(f"Books with ≥ {MIN_BOOK_RATINGS} ratings: {len(popular_books)}")

# Select top N
top_books_raw = popular_books.head(TOP_N_BOOKS)
top_isbns_all = top_books_raw.index.tolist()

existing_isbns = set(deduplicated_books["ISBN"])
top_isbns_matched = [isbn for isbn in top_isbns_all if isbn in existing_isbns]
print(f"[DEBUG] Found {len(top_isbns_matched)} books that exist in books.csv after deduplication.")

ratings_filtered = ratings[ratings["ISBN"].isin(top_isbns_matched)]
books_filtered = deduplicated_books[deduplicated_books["ISBN"].isin(top_isbns_matched)]

print(f"Filtered ratings: {ratings_filtered.shape}, Filtered books: {books_filtered.shape}")
print(f"Filtered books:\n{books_filtered[['ISBN', 'Book-Title']].head()}")

books_filtered.to_csv("filtered_books.csv", index=False)
ratings_filtered.to_csv("filtered_ratings.csv", index=False)
print("Saved filtered_books.csv and filtered_ratings.csv")

user_item_matrix = ratings_filtered.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')
print(f"User-item matrix shape: {user_item_matrix.shape}")
print(f"Sample rows:\n{user_item_matrix.head(3)}")

user_item_matrix.to_csv("user_item_matrix.csv")
print("Saved user_item_matrix.csv")

