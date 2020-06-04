# Sentiment Analysis
import pandas as pd
import random
import nltk
import numpy as np
nltk.download('vader_lexicon')
from plotly.offline import init_notebook_mode, iplot
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import classify
from nltk import NaiveBayesClassifier

# Changing Vader - to add Covid
new_words = {
    'covid': -5,
    'covid-19': -5,
    'COVID': -5,
    'COVID-19': -5
}

df = pd.read_csv('/Users/ani1705/Downloads/College-Plans_withText8.csv')
df.head()
# For text5 remove
df = df.drop(df.columns[[1,2,3,4]], axis=1)
df.columns
df.dropna(subset = ['text'], inplace=True)
df = df[df['text_length'] > 50]
df.reset_index(drop=True, inplace=True)
df

df['text'] = df['text'].astype(str)


def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(new_words)
    scores = sid.polarity_scores(text)
    return(scores['compound'])


df['Sent_Score'] = df['text'].apply(get_sentiment)
df['Sentiment'] = ''

df

for i in range(df['Sent_Score'].shape[0]):
    if df['Sent_Score'][i] >= 0.1:
        df['Sentiment'][i] = 'Positive'
    elif df['Sent_Score'][i] <= -0.1:
        df['Sentiment'][i] = 'Negative'
    else:
        df['Sentiment'][i] = 'Neutral'

df
# Uploading it
df.to_csv('/Users/ani1705/Desktop/College-Plans_withSentiment.csv')
df.columns
df.plan.value_counts()
df2 = df[df['plan'] == 'Planning for online']
df2.Sentiment.value_counts()
np.average(df2.Sent_Score)

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

df2['Sent_Score'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')


# Visually make it interesting, and using ML to make the model more accurate
# visualization in terms of maps
#
positive_data = df[df['Sentiment'] == 'Positive']
negative_data = df[df['Sentiment'] == 'Negative']

positive_data.reset_index(drop=True, inplace=True)
negative_data.reset_index(drop=True, inplace=True)

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_data)
negative_tokens_for_model = get_tweets_for_model(negative_data)
dataset = {}
keys = [positive_data['text'][i] for i in range(positive_data.shape[0])]
values = [positive_data['Sentiment'][i] for i in range(positive_data.shape[0])]
positive_dataset = dict(zip(keys, values))

dataset = {}
keys = [negative_data['text'][i] for i in range(negative_data.shape[0])]
values = [negative_data['Sentiment'][i] for i in range(negative_data.shape[0])]
negative_dataset = dict(zip(keys, values))


positive = [(keys, "Positive") for keys in positive_dataset]

negative = [(keys, "Negative") for keys in negative_dataset]

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive)
negative_tokens_for_model = get_tweets_for_model(negative)

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:429]
test_data = dataset[429:]


classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))







# Initialize vader_lexicon
sid = SentimentIntensityAnalyzer()

sid.lexicon.update(new_words)

# text
message_text = df['text'].str.cat()

print(message_text)
# Calling the polarity_scores method on sid and passing in the
# message_text outputs a dictionary with negative, neutral, positive,
# and compound scores for the input text
scores = sid.polarity_scores(message_text)

# Now conducting sentence level Analysis
import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image

stopwords = set(STOPWORDS)
stopwords.update(["the", "to", "on","and", "for", "will", "the fall"])

wordcloud = WordCloud(stopwords = stopwords, max_font_size=50, max_words=150, background_color="white").generate(message_text)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



# The tokenize method breaks up the paragraph into a list of strings. In this example, note that the tokenizer is confused by the absence of spaces after periods and actually fails to break up sentences in two instances. How might you fix that?

tokenized_sentence = nltk.word_tokenize(message_text)

# Split it by categories we have already
# t-tests for categories, to see if there are difference in means




# We add the additional step of iterating through the list of sentences and calculating and printing polarity scores for each one.

pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

for word in tokenized_sentence:
    if (sid.polarity_scores(word)['compound']) >= 0.1:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['compound']) <= -0.1:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)
print('Positive:',pos_word_list)
print('Neutral:',neu_word_list)
print('Negative:',neg_word_list)
import collections
counterneg=collections.Counter(neg_word_list)
print(counterneg)
counterpos = collections.Counter(pos_word_list)
print(counterpos)
counterpos = counterpos.most_common(10)
counterpos = dict(counterpos)
counterneg = counterneg.most_common(10)
counterneg = dict(counterneg)
counterneg.values()
for key in counterneg:
    if counterneg[key] > 0:
        counterneg[key] = -counterneg[key]

import matplotlib.pyplot as plt
%%matplotlib inline
import itertools

x = {**counterpos, **counterneg}

x = list(counterpos.keys())
y = list(counterpos.values())
x2 = list(counterneg.keys())
y2 = list(counterneg.values())
x = x + x2
y = y + y2
df = pd.DataFrame.from_dict(x, orient='index')
df.rename(columns = {0: 'hero'}, inplace=True)
df['colors'] = 'r'
df.loc[df['hero']>=0,'colors'] = 'b'
df = df.sort_values(by='hero')
plt.barh(df.index,df.hero, align='center', alpha=0.4, color = df.colors)
plt.xlabel('Frequency')
plt.title('Top 10 positive and negative words used by Colleges')
plt.show()
plt.savefig('foo.png', bbox_inches='tight')





score = sid.polarity_scores(message_text)
print('\nScores:', score)
