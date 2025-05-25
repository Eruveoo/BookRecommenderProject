import streamlit as st
import pandas as pd
from knn_recommender import generate_book_recommendations

st.set_page_config(page_title="Book Recommender", layout="wide", initial_sidebar_state="collapsed")

@st.cache_data
def load_data():
    try:
        books_df = pd.read_csv("filtered_books.csv", dtype=str).dropna(subset=['ISBN', 'Book-Title'])
        user_item_matrix_df = pd.read_csv("user_item_matrix.csv", index_col=0)
    except FileNotFoundError:
        st.error("🚨 Critical Error: 'filtered_books.csv' or 'user_item_matrix.csv' not found. "
                 "Please ensure these files are in the same directory as app1.py.")
        st.stop()  # Stop execution if files are missing

    book_info_dict = books_df.set_index("ISBN")[["Book-Title", "Book-Author", "Publisher"]].to_dict(orient="index")

    books_df['DisplayString'] = books_df['Book-Title'] + " by " + books_df['Book-Author'].fillna('Unknown Author')

    title_author_to_isbn_map = books_df.drop_duplicates(subset=['DisplayString']).set_index('DisplayString')[
        'ISBN'].to_dict()

    display_strings_for_selection = sorted(list(title_author_to_isbn_map.keys()))

    return user_item_matrix_df, book_info_dict, title_author_to_isbn_map, display_strings_for_selection, books_df


user_item_matrix, book_info, title_author_to_isbn, book_display_strings, books_data_full = load_data()

st.title("Book Recommender")
st.markdown("Select one or more books you've enjoyed, and we'll suggest what to read next!")

st.header("Your Liked Books:")
selected_display_strings = st.multiselect(
    label="Search and select books you've enjoyed:",
    options=book_display_strings,
    placeholder="Type to search for a book title..."
)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_books_cache' not in st.session_state:
    st.session_state.selected_books_cache = []

col1, col2 = st.columns(2)
if col1.button("📚 Get Recommendations", type="primary", use_container_width=True):
    if selected_display_strings:
        liked_isbns = [title_author_to_isbn[title_auth] for title_auth in selected_display_strings if
                       title_auth in title_author_to_isbn]

        if not liked_isbns:
            st.warning("Could not map selected books to ISBNs. Please try again.")
            st.session_state.recommendations = []
        else:
            st.session_state.selected_books_cache = selected_display_strings
            with st.spinner("⏳ Finding books you might love..."):
                recommendations = generate_book_recommendations(
                    liked_isbns,
                    user_item_matrix,
                    book_info,
                    num_recs=6
                )
                st.session_state.recommendations = recommendations
    else:
        st.warning("Please select at least one book to get recommendations.")
        st.session_state.recommendations = None

if col2.button("Clear Selections", use_container_width=True):
    st.session_state.recommendations = None
    st.session_state.selected_books_cache = []
    st.rerun()

if st.session_state.selected_books_cache:
    st.subheader(f"Recommendations based on your love for:")
    liked_titles_display = "<ul>"
    for book_str in st.session_state.selected_books_cache:
        liked_titles_display += f"<li><em>{book_str}</em></li>"
    liked_titles_display += "</ul>"
    st.markdown(liked_titles_display, unsafe_allow_html=True)
    st.markdown("---")

if st.session_state.recommendations is not None:
    if st.session_state.recommendations:
        st.subheader("Here are some books you might enjoy:")

        num_recs = len(st.session_state.recommendations)
        cols = st.columns(min(num_recs, 3))

        for i, rec in enumerate(st.session_state.recommendations):
            with cols[i % min(num_recs, 3)]:
                with st.container(border=True):
                    st.markdown(f"##### {i + 1}. {rec['title']}")
                    st.caption(f"_{rec['author']}_")
                    st.markdown(f"<small>ISBN: {rec['isbn']}</small>", unsafe_allow_html=True)
                    st.markdown(f"<small>Score: {rec['score']:.4f}</small>", unsafe_allow_html=True)

    elif st.session_state.selected_books_cache:
        st.info("🤔 We couldn't find any new recommendations based on your current selection. "
                "Try adding more or different books!")

st.markdown("---")
st.markdown("<small>Book Recommender v0.1</small>", unsafe_allow_html=True)