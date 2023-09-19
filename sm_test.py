
# test code

import pandas as pd
import flair
from textblob import TextBlob
import os
import datetime
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

fmt = '%Y-%m-%d'


def get_sentiment_val_for_flair(sentiments):
    """
    parse input of the format [NEGATIVE (0.9284018874168396)] and return +ve or -ve float value
    :param sentiments:
    :return:
    """
    total_sentiment = str(sentiments)
    neg = 'NEGATIVE' in total_sentiment
    if neg:
        total_sentiment = total_sentiment.replace('NEGATIVE', '')
    else:
        total_sentiment = total_sentiment.replace('POSITIVE', '')

    total_sentiment = total_sentiment.replace('(', '').replace('[', '').replace(')', '').replace(']', '')

    val = float(total_sentiment)
    if neg:
        return -val
    return val

