# Unsupervised learning


## Dataset

We were given a subset of [Carnegie Mellon Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/), that contains 42,306 movie plot summaries extracted from Wikipedia and aligned metadata extracted from Freebase, including the movie title, one or more labels about to genre and a short summary. 


## Movie recommendation system based on the content

In the first part, we implement a content based recommendation system for movies. The purpose of this system is to recommend movies that might interest the user, using content-based filtering method. First, we convert each summary in a tf-idf vector representation and, then, we recomment to the user movies that have similar representation (accoding to cosine similarity) with those movies that he have already seen.


### Pre-processing methods used:

- Removing punctuation
- Removing names
- Stemming

### Tf-idf vectorizer parameters used:

- Removing stop words
- Removing words with high df (document frequency)
- Removing word with low df
- Use bigrams along with unigrams

### Some recommendations

```
Target movie
Id: 4
Title: Star Trek: Insurrection
Categories: "Thriller",  "Science Fiction",  "Adventure",  "Drama",  "Romance Film",  "Action"

Recommended movie 1
Id: 3287
Title: Star Trek IV: The Voyage Home
Categories: "Time travel",  "Science Fiction",  "Family Film",  "Adventure",  "Comedy"

Recommended movie 2
Id: 4271
Title: Women of the Prehistoric Planet
Categories: "Science Fiction",  "Action",  "Adventure"

Recommended movie 3
Id: 2352
Title: Galaxy of Terror
Categories: "Science Fiction",  "Action",  "Horror",  "Sci-Fi Horror"

Recommended movie 4
Id: 84
Title: Dark Star
Categories: "Parody",  "Thriller",  "Science Fiction",  "Indie",  "Cult",  "Drama",  "Comedy",  "Adventure"

Recommended movie 5
Id: 3600
Title: Pandorum
Categories: "Thriller",  "Science Fiction",  "Horror",  "Indie",  "Sci-Fi Horror",  "Action"
```

## Somoclu - Self Organizing Maps¶

In the second part, we rely on the topological properties of the Self Organizing Maps (SOM) in order to create a 2-dimensional grid in which we project the movies of our dataset based on their content but most importantly their genre.

### Results

- U-matrix of best model



- Distribution of categories in some clusters of best model



