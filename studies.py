from utils import count_tweets, process_tweet
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import nltk
from formulas import train_naive_bayes, naive_bayes_predict, test_naive_bayes, get_ratio, get_words_by_threshold

nltk.download('stopwords')
nltk.download('twitter_samples')

# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

# Build the freqs dictionary
freqs = count_tweets({}, train_x, train_y)

# check logprior (0.0) and loglikelihood (9089)
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print('logprior:', logprior)
print('loglikelihood:', len(loglikelihood))

# Testing prediciton with a custom tweet - The expected output is around 1.57 (positive sentiment)
my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The output is', p)

# Expected Accuracy: 0.9980
print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

# predict other tweets. The expected output is:
## I am happy -> 2.15
## I am bad -> -1.29
## this movie should have been great. -> 2.14
## great -> 2.14
## great great -> 4.28
## great great great -> 6.41
## great great great great -> 8.55

for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
#     print(f'{tweet} -> {p:.2f} ({p_category})')
    print(f'{tweet} -> {p:.2f}')

# Check the sentiment of another custom tweet
my_tweet = 'you are great'
print('custom tweet result:', naive_bayes_predict(my_tweet, logprior, loglikelihood))

# check ratio
print('ratio (happi): ', get_ratio(freqs, 'happi'))

# find negative words at or below a threshold
print(get_words_by_threshold(freqs, label=0, threshold=0.05))
# find positive words at or above a threshold
### Notice the difference between the positive and negative ratios.
### Emojis like :( and words like 'me' tend to have a negative connotation.
#### Other words like 'glad', 'community', and 'arrives' tend to be found in the positive tweets.
get_words_by_threshold(freqs, label=1, threshold=10)

# Some error analysis
print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            process_tweet(x)).encode('ascii', 'ignore')))