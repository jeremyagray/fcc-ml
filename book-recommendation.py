#!/usr/bin/env python

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import urllib
import zipfile

# Temporary directory paths.
tmp = '/home/gray/tmp'
directory = 'fcc-ml'
tmpdir = os.path.join(tmp, directory)

# Test for the data files before downloading them.
books = os.path.join(tmpdir, 'BX-Books.csv')
ratings = os.path.join(tmpdir, 'BX-Book-Ratings.csv')

if not (os.path.isfile(books) and os.path.isfile(ratings)):
    # Create the temporary directory if it does not exist.
    if os.exists(tmp):
        if not os.exists(tmpdir):
            os.makedirs(tmpdir)

    # Download and prepare data set.
    url = 'https://cdn.freecodecamp.org/project-data/books/book-crossings.zip'

    # Make a good fake request for the CDN.
    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})

    # Download the file and save it.
    data = os.path.join(tmpdir, 'book-crossings.zip')
    with urllib.request.urlopen(req) as response, open(data, 'wb') as file:
        shutil.copyfileobj(response, file)

    with zipfile.ZipFile(data, 'r') as archive:
        archive.extractall(tmpdir)

# Load CSV data into pandas.
df_books = pd.read_csv(
    books,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

exit()

# Return recommended books.
def get_recommends(book=""):
    pass
    # return recommended_books


books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)


def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is "
                                "(Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False

    recommended_books = [
        "I'll Be Seeing You",
        'The Weight of Water',
        'The Surgeon',
        'I Know This Much Is True'
    ]

    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]

    for i in range(2):
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False

    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You havn't passed yet. Keep trying!")

    return


test_book_recommendation()
