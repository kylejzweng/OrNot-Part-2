import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from nltk.collocations import BigramCollocationFinder  
from nltk.collocations import TrigramCollocationFinder 
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder
from nltk.text import ConcordanceIndex

df = pd.read_csv('full_ornot_dataset.csv')

# Dropping any remaining null headers
df.dropna(subset = ['headers'], inplace = True)
# print(df.info())

# Make headers and reviews lowercase
df['headers'] = df['headers'].str.lower()
df['body'] = df['body'].str.lower()

# Tokenization - splitting headers and reviews into lists of words. Also removing any punctuation
tokenizer = RegexpTokenizer("[\w]+")
df['tokenized_headers'] = df['headers'].map(tokenizer.tokenize)
df['tokenized_body'] = df['body'].map(tokenizer.tokenize)

# Removing stop words
stop_words = set(stopwords.words('english'))

df['tokenized_nostop_headers'] = df['tokenized_headers'].apply(lambda x: [word for word in x if word not in stop_words])
df['tokenized_nostop_body'] = df['tokenized_body'].apply(lambda x: [word for word in x if word not in stop_words])

# Most common words in headers
# Flattening so that we have one big list instead of rows of lists
flattened_headers = []
for sublist in df['tokenized_nostop_headers']:
    for word in sublist:
        flattened_headers.append(word)
    
header_word_count = {}
for word in flattened_headers:
    if word not in header_word_count:
        header_word_count[word] = 1
    else:
        header_word_count[word] = header_word_count[word] + 1
    
sorted_header_word_count = sorted(header_word_count.items(), key = lambda x: x[1], reverse = True)
# print(sorted_header_word_count)

# Making wordcloud with most used words
wordcloud = WordCloud(width = 500, height = 500).generate_from_frequencies(header_word_count)
plt.imshow(wordcloud)
plt.axis('off')
# plt.show()

# Making new wordcloud excluding any product words like jersey, shorts, bibs, house etc so it's just adjectives
top_words_header = dict(sorted_header_word_count[:100])
words_to_exclude_header = ['jersey', 'shorts', 'bibs', 'house', 'bib', 'jacket', 'vest', 'ornot', 'short', 'ever', 'shirt', 'mission', 'kit', 'long', 'weather', 'bag', 'product', 'go', 'wind', 'like', 'review', 'layer', 'piece', 'one', 'climate', 'fit', 'pants', 'pair', 'winter', 'merino', 'bike', 'really', 'high', 'shell', 'sleeve', 'jerseys', 'size', 'another', 'blue', 'cycling', 'well', 'little', 'work', 'thermal', 'buy', 'summer', 'small', 'cargo', 'orange', 'first', 'stone', 'socks', 'wear', 'base', 'bought', 'looking' 'ls', 'made', 'grid', 'gloves', 'pockets', 'tight', 'ride', 'colors', 'purchase', 'riding', 'sweatshirt', 'sizing']
filtered_header_word_count = {word: count for word, count in top_words_header.items() if word not in words_to_exclude_header}
wordcloud_header = WordCloud(width=300, height=300).generate_from_frequencies(filtered_header_word_count)
plt.imshow(wordcloud_header)
plt.axis('off')
# plt.show()

# Most common words in reviews
# Flattening reviews so that we have one big list instead of rows of lists
flattened_reviews = []
for sublist in df['tokenized_nostop_body']:
    for word in sublist:
        flattened_reviews.append(word)

# Also flattening headers in case of later use
flattened_headers = []
for sublist in df['tokenized_nostop_headers']:
    for word in sublist:
        flattened_headers.append(word)
    
review_word_count = {}
for word in flattened_reviews:
    if word not in review_word_count:
        review_word_count[word] = 1
    else:
        review_word_count[word] = review_word_count[word] + 1
    
sorted_review_word_count = sorted(review_word_count.items(), key = lambda x: x[1], reverse = True)
# print(sorted_review_word_count)

# Making wordcloud with most used words
wordcloud = WordCloud(width = 500, height = 500).generate_from_frequencies(review_word_count)
plt.imshow(wordcloud)
plt.axis('off')
# plt.show()

# Making new wordcloud with just adjectives that feel relevent
top_words_body = dict(sorted_review_word_count[:100])
words_to_exclude_body = ['jersey', 'ornot', 'shorts', 'bibs', 'jacket', 'wear', 'really', 'fit', 'would', 'long', 'ride', 'one', 'small', 'rides', 'medium', 'bit', 'bike', 'pair', 'vest', 'layer', "i've", 'get', 'little', 'also', 'riding', 'enough', 'house', 'tight', 'pocket', 'first', 'back', 'much', 'large', 'look', 'right', 'short', 'made', 'bib', 'jerseys', 'around', '5', 'sleeve', 'weather', 'wind', 'even', 'bought', 'wearing', 'snug', 'sleeves', 'days', 'go', 'base', 'still', 'day', 'way', 'far', 'feels', 'got', 'time', 'definitely', 'looking', 'worn', 'length', 'without', 'use', 'could', 'used', 'buy', 'stretch', 'cold', '6', 'two']
filtered_body_word_count = {word: count for word, count in top_words_body.items() if word not in words_to_exclude_body}
review_wordcloud = WordCloud(width=300, height=300).generate_from_frequencies(filtered_body_word_count)
plt.imshow(review_wordcloud)
plt.axis('off')
# plt.show()

# VADER Sentiment Scores for headers and reviews

sia = SentimentIntensityAnalyzer()

df['headers_sentiment_VADER'] = df['tokenized_nostop_headers'].apply(lambda x: sia.polarity_scores(' '.join(x)))

df['headers_VADER_neg'] = df['headers_sentiment_VADER'].apply(lambda x: x['neg'])
df['headers_VADER_neu'] = df['headers_sentiment_VADER'].apply(lambda x: x['neu'])
df['headers_VADER_pos'] = df['headers_sentiment_VADER'].apply(lambda x: x['pos'])
df['headers_VADER_compound'] = df['headers_sentiment_VADER'].apply(lambda x: x['compound'])

df['reviews_sentiment_VADER'] = df['tokenized_nostop_body'].apply(lambda x: sia.polarity_scores(' '.join(x)))

df['reviews_VADER_neg'] = df['reviews_sentiment_VADER'].apply(lambda x: x['neg'])
df['reviews_VADER_neu'] = df['reviews_sentiment_VADER'].apply(lambda x: x['neu'])
df['reviews_VADER_pos'] = df['reviews_sentiment_VADER'].apply(lambda x: x['pos'])
df['reviews_VADER_compound'] = df['reviews_sentiment_VADER'].apply(lambda x: x['compound'])

# Putting VADER compound scores into buckets to see which compound scores are the most common
bin_edges = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
df['headers_VADER_compound_buckets'] = pd.cut(df['headers_VADER_compound'], bins = bin_edges)
df['reviews_VADER_compound_buckets'] = pd.cut(df['reviews_VADER_compound'], bins = bin_edges)

header_compound_count = df['headers_VADER_compound_buckets'].value_counts()
header_compound_count = header_compound_count.reindex(bin_edges)
review_compound_count = df['reviews_VADER_compound_buckets'].value_counts()
review_compound_count = review_compound_count.reindex(bin_edges)

plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
header_compound_count.plot(kind = 'bar')
plt.title('Headers')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
review_compound_count.plot(kind = 'bar')
plt.title('Reviews')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.suptitle('Distrubtion of VADER Compound Scores')
plt.tight_layout()
# plt.show()

# Checking to see how the compound, pos, and neg scores compare to the star ratings for the headers
plt.figure(figsize=(15, 8))

# Pos VADER scores
plt.subplot(2, 3, 1)
sns.barplot(data = df,
            x = 'star',
            y = 'headers_VADER_pos',
            color = 'limegreen')
plt.title('Positive Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')

# Neutral VADER scores
plt.subplot(2, 3, 2)
sns.barplot(data = df,
            x = 'star',
            y = 'headers_VADER_neu',
            color = 'khaki')
plt.title('Neutral Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')

# Neg VADER scores
plt.subplot(2, 3, 3)
sns.barplot(data = df,
            x = 'star',
            y = 'headers_VADER_neg',
            color = 'firebrick')
plt.title('Negative Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')

# Compound VADER scores
plt.subplot(2, 3, 5)
sns.barplot(data = df,
            x = 'star',
            y = 'headers_VADER_compound', 
            color = 'royalblue')
plt.title('Compound Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')
plt.suptitle('VADER Sentiment Scores by Star Rating for Headers')
plt.tight_layout()
# plt.show()

# Checking to see how the compound, pos, and neg scores compare to the star ratings for the reviews
plt.figure(figsize=(15, 8))

# Pos VADER scores
plt.subplot(2, 3, 1)
sns.barplot(data = df,
            x = 'star',
            y = 'reviews_VADER_pos',
            color = 'limegreen')
plt.title('Positive Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')

# Neutral VADER scores
plt.subplot(2, 3, 2)
sns.barplot(data = df,
            x = 'star',
            y = 'reviews_VADER_neu',
            color = 'khaki')
plt.title('Neutral Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')

# Neg VADER scores
plt.subplot(2, 3, 3)
sns.barplot(data = df,
            x = 'star',
            y = 'reviews_VADER_neg',
            color = 'firebrick')
plt.title('Negative Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')

# Compound VADER scores
plt.subplot(2, 3, 5)
sns.barplot(data = df,
            x = 'star',
            y = 'reviews_VADER_compound', 
            color = 'royalblue')
plt.title('Compound Sentiment')
plt.xlabel('Star Rating')
plt.ylabel('Sentiment Score')
plt.suptitle('VADER Sentiment Scores by Star Rating for Reviews')
plt.tight_layout()
# plt.show()

# Relationship between VADER compound score for headers vs review
plt.figure(figsize = (12, 8))
sns.scatterplot(data = df,
                x = 'reviews_VADER_compound',
                y = 'headers_VADER_compound')
plt.title('Relationship between Header and Review VADER Compound Scores')
plt.xlabel('Reviews Compound Score')
plt.ylabel('Headers Compound Score')
# plt.show()  # Basically no relationship which is weird

# Hunch that compound score is affected by length of review/header. Finding length of each review
df['length_of_review'] = df['tokenized_nostop_body'].str.len()
df['length_of_header'] = df['tokenized_nostop_headers'].str.len()

# print('Avg length of reviews: ', df['length_of_review'].mean())     # 24.25 words
# print('Avg length of headers: ', df['length_of_header'].mean())     # 2.7
# A tweet 280 characters
# One word is typically about 4.7 characters

plt.figure(figsize = (12, 8))

plt.subplot(2, 3, 1)
sns.scatterplot(data = df,
                x = 'reviews_VADER_pos',
                y = 'length_of_review',
                color = 'limegreen')
plt.title('Positive Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Length of Review')

plt.subplot(2, 3, 2)
sns.scatterplot(data = df,
                x = 'reviews_VADER_neu',
                y = 'length_of_review',
                color = 'khaki')
plt.title('Neutral Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Length of Review')

plt.subplot(2, 3, 3)
sns.scatterplot(data = df,
                x = 'reviews_VADER_neg',
                y = 'length_of_review',
                color = 'firebrick')
plt.title('Negative Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Length of Review')

plt.subplot(2, 3, 5)
sns.scatterplot(data = df,
                x = 'reviews_VADER_compound',
                y = 'length_of_review',
                color = 'royalblue')
plt.title('Compound Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Length of Review')
plt.suptitle('VADER Sentiment Scores by Length of Reviews')
plt.tight_layout()
# plt.show()

# Average pos, neu, neg, and comp scores for headers and reviews
average_header_pos_score = df['headers_VADER_pos'].mean()
average_header_neu_score = df['headers_VADER_neu'].mean()
average_header_neg_score = df['headers_VADER_neg'].mean()
average_header_comp_score = df['headers_VADER_compound'].mean()

average_review_pos_score = df['reviews_VADER_pos'].mean()
average_review_neu_score = df['reviews_VADER_neu'].mean()
average_review_neg_score = df['reviews_VADER_neg'].mean()
average_review_comp_score = df['reviews_VADER_compound'].mean()

# print('Headers:')
# print('Avg Positive Score: ', average_header_pos_score)
# print('Avg Neutral Score: ', average_header_neu_score)
# print('Avg Negative Score: ', average_header_neg_score)
# print('Avg Compound Score: ', average_header_comp_score)
# print('Reviews:')
# print('Avg Positive Score: ', average_review_pos_score)
# print('Avg Neutral Score: ', average_review_neu_score)
# print('Avg Negative Score: ', average_review_neg_score)
# print('Avg Compound Score: ', average_review_comp_score)

# Avg compound score for reviews and headers by year
average_review_comp_score_by_year = df.groupby('year')['reviews_VADER_compound'].mean().sort_index()
review_count_by_year = df['year'].value_counts().sort_index()
average_header_comp_score_by_year = df.groupby('year')['headers_VADER_compound'].mean()
header_count_by_year = df['year'].value_counts().sort_index()

# How has compound score changed over the years for reviews?
fig, ax = plt.subplots(figsize=(10, 6))

average_review_comp_score_by_year.plot(kind='bar', ax=ax, color='blue', label='Avg Review Compound Score')

years_range = range(len(review_count_by_year))

ax2 = ax.twinx()  
ax2.plot(years_range, review_count_by_year, color='black', marker='o', label='Review Count')

ax2.set_xticks(years_range)
ax2.set_xticklabels(review_count_by_year.index)
ax.set_xlabel('Year')
ax.set_ylabel('Avg Review Compound Score', color='blue')
ax2.set_ylabel('Review Count', color='black')
ax.set_title('Average Review Compound Score and Review Count by Year')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')
# plt.show()

# 5 lowest review compound scores for 5 star rating
five_star_reviews = df[df['star'] == 5]
lowest_review_comp_score_and_5_star = five_star_reviews.nsmallest(5, 'reviews_VADER_compound')

# for index, review in lowest_review_comp_score_and_5_star.iterrows():
#     print("User:", review['users'])
#     print("Date:", review['date'])
#     print("Product:", review['product'])
#     print("Headers:", review['headers'])
#     print("Body:", review['body'])
#     print("Star Rating:", review['star'])
#     print("VADER Compound Score (Review):", review['reviews_VADER_compound'])
#     print("VADER Compound Score (Header):", review['headers_VADER_compound'])
#     print("============================================")

# lowest_comp_header_score_and_5_star = five_star_reviews.nsmallest(5, 'headers_VADER_compound')
# for index, review in lowest_comp_header_score_and_5_star.iterrows():
#     print("User:", review['users'])
#     print("Date:", review['date'])
#     print("Product:", review['product'])
#     print("Headers:", review['headers'])
#     print("Body:", review['body'])
#     print("Star Rating:", review['star'])
#     print("VADER Compound Score (Header):", review['headers_VADER_compound'])
#     print("VADER Compound Score (Review):", review['reviews_VADER_compound'])
#     print("============================================")

# 5 highest review compound scores for 1 star rating
one_star_reviews = df[df['star'] == 1]
highest_review_comp_score_and_1_star = one_star_reviews.nlargest(5, 'reviews_VADER_compound')

# for index, review in highest_review_comp_score_and_1_star.iterrows():
#     print("User:", review['users'])
#     print("Date:", review['date'])
#     print("Product:", review['product'])
#     print("Headers:", review['headers'])
#     print("Body:", review['body'])
#     print("Star Rating:", review['star'])
#     print("VADER Compound Score (Review):", review['reviews_VADER_compound'])
#     print("VADER Compound Score (Header):", review['headers_VADER_compound'])
#     print("============================================")

highest_header_comp_score_and_1_star = one_star_reviews.nlargest(5, 'headers_VADER_compound')

# for index, review in highest_header_comp_score_and_1_star.iterrows():
#     print("User:", review['users'])
#     print("Date:", review['date'])
#     print("Product:", review['product'])
#     print("Headers:", review['headers'])
#     print("Body:", review['body'])
#     print("Star Rating:", review['star'])
#     print("VADER Compound Score (Header):", review['headers_VADER_compound'])
#     print("VADER Compound Score (Review):", review['reviews_VADER_compound'])
#     print("============================================")

# Star ratings across product categories was basically all identical. What about compound scores?
average_review_comp_score_by_product_category = df.groupby('product_category')['reviews_VADER_compound'].mean()
# print(average_review_comp_score_by_product_category)

plt.figure(figsize = (12, 8))

average_review_comp_score_by_product_category.plot(kind = 'bar')
plt.title('Average VADER Compound Score by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Compound Score')
plt.xticks(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# Finding common two-word phrases throughout reviews
bigram_finder = BigramCollocationFinder.from_words(flattened_reviews)
bigram_finder.apply_freq_filter(50)
bigrams_with_freq_reviews = bigram_finder.ngram_fd.items()
sorted_bigrams_with_freq_reviews = sorted(bigrams_with_freq_reviews, key = lambda x: x[1], reverse = True)
# for bigram, freq in sorted_bigrams_with_freq_reviews:
#     print(f": {bigram}, {freq}")

# Finding common three-word phrases throughout reviews
trigram_finder = TrigramCollocationFinder.from_words(flattened_reviews)
trigram_finder.apply_freq_filter(10)
trigrams_with_freq_reviews = trigram_finder.ngram_fd.items()
sorted_trigrams_with_freq_reviews = sorted(trigrams_with_freq_reviews, key = lambda x: x[1], reverse = True)
# for trigram, freq in sorted_trigrams_with_freq_reviews:
#     print(f": {trigram}, {freq}")

# Finding common four-word phrases throughout reviews
quadgram_finder = QuadgramCollocationFinder.from_words(flattened_reviews)
quadgram_finder.apply_freq_filter(5)
quadgrams_with_freq_reviews = quadgram_finder.ngram_fd.items()
sorted_quadgrams_with_freq_reviews = sorted(quadgrams_with_freq_reviews, key = lambda x: x[1], reverse = True)
# for quadgram, freq in sorted_quadgrams_with_freq_reviews:
#     print(f": {quadgram}, {freq}")

# Finding common two-word phrases throughout headers
bigram_finder = BigramCollocationFinder.from_words(flattened_headers)
bigram_finder.apply_freq_filter(30)
bigrams_with_freq_headers = bigram_finder.ngram_fd.items()
sorted_bigrams_with_freq_headers = sorted(bigrams_with_freq_headers, key = lambda x: x[1], reverse = True)
# for bigram, freq in sorted_bigrams_with_freq_headers:
#     print(f": {bigram}, {freq}")

# Finding common three-word phrases throughout headers
trigram_finder = TrigramCollocationFinder.from_words(flattened_headers)
trigram_finder.apply_freq_filter(15)
trigrams_with_freq_headers = trigram_finder.ngram_fd.items()
sorted_trigrams_with_freq_headers = sorted(trigrams_with_freq_headers, key = lambda x: x[1], reverse = True)
# for trigram, freq in sorted_trigrams_with_freq_headers:
#     print(f": {trigram}, {freq}")

# Finding common four-word phrases throughout headers
quadgram_finder = QuadgramCollocationFinder.from_words(flattened_headers)
quadgram_finder.apply_freq_filter(5)
quadgrams_with_freq_headers = quadgram_finder.ngram_fd.items()
sorted_quadgrams_with_freq_headers = sorted(quadgrams_with_freq_headers, key = lambda x: x[1], reverse = True)
# for quadgram, freq in sorted_quadgrams_with_freq_headers:
#     print(f": {quadgram}, {freq}")