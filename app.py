import pandas as pd
import streamlit as st
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv("books.csv", on_bad_lines='skip')
df.drop(['bookID', 'isbn', 'isbn13'], axis=1, inplace=True)


rating_df = pd.get_dummies(df['average_rating'].apply(lambda x: "between 0 and 1" if x > 0 and x <= 1 else
                                                      "between 1 and 2" if x > 1 and x <= 2 else
                                                      "between 2 and 3" if x > 2 and x <= 3 else
                                                      "between 3 and 4" if x > 3 and x <= 4 else
                                                      "between 4 and 5"))

language_df = pd.get_dummies(df['language_code'])

features = pd.concat([rating_df, language_df, df[['average_rating', 'ratings_count']]], axis=1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)


model = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)


def recommend_books_publishers(publisher_name):
    recommended_books = df[df['publisher'] == publisher_name][['title', 'average_rating']]
    recommended_books = recommended_books.sort_values(by='average_rating', ascending=False).head(10)
    return recommended_books

def recommend_books_authors(authors_name):
    recommended_books = df[df['authors'] == authors_name][['title', 'average_rating']]
    recommended_books = recommended_books.sort_values(by='average_rating', ascending=False).head(10)
    return recommended_books

def recommend_books_lang(language):
    recommended_books = df[df['language_code'] == language][['title', 'average_rating']]
    recommended_books = recommended_books.sort_values(by='average_rating', ascending=False).head(10)
    return recommended_books


def BookRecommender(book_name):
    book_list_name = []
    book_id = df[df['title'] == book_name].index[0]
    for newid in idlist[book_id]:
        book_list_name.append(df.iloc[newid].title)
    return book_list_name


st.title("BOOK RECOMMENDATION SYSTEM")


st.sidebar.title("Recommendations")


selected_publisher = st.sidebar.selectbox("Select Publisher", df['publisher'].value_counts().index)
selected_author = st.sidebar.selectbox("Select Author", df['authors'].value_counts().index)
selected_language = st.sidebar.selectbox("Select Language", df['language_code'].value_counts().index)


st.subheader("Recommendations by Publisher")
st.write(recommend_books_publishers(selected_publisher))

st.subheader("Recommendations by Author")
st.write(recommend_books_authors(selected_author))

st.subheader("Recommendations by Language")
st.write(recommend_books_lang(selected_language))


book_to_recommend = st.sidebar.selectbox("Select Book for Recommendation", df['title'].value_counts().index)
recommended_books = BookRecommender(book_to_recommend)
st.subheader("Books Recommended for '{}'".format(book_to_recommend))
st.write(recommended_books)
