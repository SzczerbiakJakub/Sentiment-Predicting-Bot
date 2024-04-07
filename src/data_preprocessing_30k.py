import pandas as pd
from text_preprocessing import TextPreprocessing as prep

df = pd.read_csv(r"..\datasets\30_k_tweets\train.csv", encoding='windows-1252')
df2 = pd.read_csv(r"..\datasets\30_k_tweets\test.csv", encoding='windows-1252')

df2 = df2.dropna()
df = pd.concat([df, df2], axis=0)

prep.get_stop_words_list()
preprocessed_list = []

for i, x in enumerate(df["text"].tolist()):
    if i%10000 == 0:
        print(i)
    preprocessed_list.append(prep.preprocess_text(str(x)))

df["text"] = preprocessed_list

neg_df = df[df["sentiment"] == "negative"]
neg_df.insert(loc=3, column='sentiment_id', value=0)

neu_df = df[df["sentiment"] == "neutral"]
neu_df.insert(loc=3, column='sentiment_id', value=1)

pos_df = df[df["sentiment"] == "positive"]
pos_df.insert(loc=3, column='sentiment_id', value=2)

resulting_df = pd.concat([neg_df, neu_df, pos_df], axis=0)
resulting_df = resulting_df.sample(frac=1, random_state=42)

resulting_df.to_csv(r".\datasets\PREPROCESSED_30k.csv", encoding="utf-8")