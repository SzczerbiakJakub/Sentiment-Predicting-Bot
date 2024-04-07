import pandas as pd
from text_preprocessing import TextPreprocessing as prep

df = pd.read_csv(r".\datasets\sentiment-emotion-labelled_Dell_tweets.csv", encoding="utf-8")
df = df.dropna()

prep.get_stop_words_list()

preprocessed_list = []
for i, x in enumerate(df["Text"].tolist()):
    if i%10000 == 0:
        print(i)
    preprocessed_list.append(prep.preprocess_text(x))

df["Text"] = preprocessed_list

neg_df = df[df["sentiment"] == "negative"]
neg_df.insert(loc=3, column='sentiment_id', value=0)
neu_df = df[df["sentiment"] == "neutral"]
neu_df.insert(loc=3, column='sentiment_id', value=1)
pos_df = df[df["sentiment"] == "positive"]
pos_df.insert(loc=3, column='sentiment_id', value=2)

resulting_df = pd.concat([neg_df, neu_df, pos_df], axis=0)
resulting_df = resulting_df.sample(frac=1, random_state=3)

resulting_df.to_csv(r".\datasets\PREPROCESED_sentiment-emotion-labelled_Dell_tweets.csv", encoding="utf-8")
