import streamlit as st
import praw
import pandas as pd
from transformers import pipeline
import datetime
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from time import sleep
from stqdm import stqdm
import streamlit.components.v1 as components

#credentials
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
    username="",
    password=""
 )


headlines = []
score = []
upvote_ratio = []
utc = []
num_comments = []
sentiment_ovl = []


st.set_page_config(page_title = "My page",layout="wide")
st.title("Michał Madej")
title = st.text_input('Subreddit name', 'politics')
st.write('The current subreddit name is:', title)

classifier = pipeline("text-classification",model='bhadresh-savani/albert-base-v2-emotion', return_all_scores=True)

@st.cache_data 
def df_loader(title):
    no_reddits = len(list(reddit.subreddit(title).hot(limit=None)))
    print("start")
    for submission in stqdm(reddit.subreddit(title).hot(limit=None), total = no_reddits):

        headlines.append(submission.title)
        score.append(submission.score)
        upvote_ratio.append(submission.upvote_ratio)

        parsed_date = datetime.utcfromtimestamp(submission.created_utc).date()
        utc.append(parsed_date)
        num_comments.append(submission.num_comments)
        prediction = classifier(submission.title)

        max_item = max(prediction[0], key=lambda x: x['score'])
        sentiment_ovl.append(max_item['label'])


    data = {'Headlines': headlines, 'Score': score,
            'Upvote Ratio': upvote_ratio, 'No_comments': num_comments,
            'Time-utc': utc, 'Sentiment': sentiment_ovl}

    df = pd.DataFrame(data)

    return df
st.sidebar.success("Select a demo above.")

df = df_loader(title)
counts = df['Sentiment'].value_counts()
st.write(counts)

count_by_date = df.groupby('Time-utc')['Headlines'].count()

count_by_sent_2 = df[df['Sentiment'] == 'anger'].groupby('Time-utc')['Sentiment'].count()
count_by_score = df.groupby('Time-utc')['Score'].sum()
print(count_by_sent_2)
st.write(count_by_sent_2)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (15,4))

ax1.bar(count_by_date.index, count_by_date.values)
ax1.set_xlabel('Data')
ax1.set_ylabel('Liczba postów')

st.pyplot(fig1)


fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize = (10,5))
ax2.pie(counts, labels = counts.index)
ax2.legend()

st.pyplot(fig2)

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize = (10,5))
ax3.bar(count_by_sent_2.index, count_by_sent_2.values)
ax3.legend()
ax3.set_xlabel('Data')
ax3.set_ylabel('Liczba postów o sentymencie anger')

st.pyplot(fig3)

st.dataframe(df)

st.write("Dodatkowe - Suma score z danej daty")

fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize = (10,5))
ax4.bar(count_by_score.index, count_by_score.values)
ax4.legend()
ax4.set_xlabel('Data')
ax4.set_ylabel('Score')

st.write(df['Score'],df['No_comments'])
st.pyplot(fig4)
st.write("Dodatkowe - Zależność score od ilości komentarzy")
fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize = (10,5))
ax5.scatter(df['Score'], df['No_comments'], s =10)
ax5.legend()
ax5.set_xlabel('Score')
ax5.set_ylabel('NO_comments')

st.pyplot(fig5)
