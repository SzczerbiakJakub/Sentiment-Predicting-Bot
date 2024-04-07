import pandas as pd
import numpy as np
from text_preprocessing import TextPreprocessing as prep



#   PREPROCESSING OF CSV DATA:

df = pd.read_csv(r".\datasets\Twitter_Data.csv", encoding="utf-8")
df = df.dropna()

print("RAW: ")
print(df["clean_text"][0])

prep.get_stop_words_list()

preprocessed_list = []
for i, x in enumerate(df["clean_text"].tolist()):
    if i%10000 == 0:
        print(i)
    preprocessed_list.append(prep.preprocess_text(x))

df["clean_text"] = preprocessed_list

print("CLEAN: ")
print(df["clean_text"][0])

texts = df.clean_text.values.tolist()

test_cat_0_df = df[df["category"] == -1]
test_cat_0_df.insert(loc=2, column='sentiment_id', value=0)
#print(test_cat_0_df[["category", "sentiment_id"]])

test_cat_1_df = df[df["category"] == 0]
test_cat_1_df.insert(loc=2, column='sentiment_id', value=1)
#print(test_cat_1_df[["category", "sentiment_id"]])

test_cat_2_df = df[df["category"] == 1]
test_cat_2_df.insert(loc=2, column='sentiment_id', value=2)
#print(test_cat_2_df[["category", "sentiment_id"]])

resulting_test_df = pd.concat([test_cat_0_df, test_cat_1_df, test_cat_2_df], axis=0)

resulting_test_df = resulting_test_df.sample(frac=1, random_state=3)

print(resulting_test_df[["clean_text", "category", "sentiment_id"]])

resulting_test_df.to_csv(".\datasets\PREPROCESSED_Twitter_Data.csv", index=True)

