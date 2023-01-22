import yfinance as yf
import streamlit as st
import pandas as pd
import pandas as pd
import requests
from yahoo_fin import news
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
import numpy as np
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
classifier=pipeline('sentiment-analysis',model=model,tokenizer=tokenizer)


API_ENDPOINT = "https://api.marketaux.com/v1/news/all"
API_TOKEN = "LhisEXuTaF5PT95IiE0dnE5LoF3rgo9SU73MqZA5"


# FUNCTIONS
def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code



# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-100):
    a = dataset[i:i+time_step, 0]
    dataX.append(a)
    dataY.append(dataset[i + time_step, 0])
  return np.array(dataX), np.array(dataY)


def return_news(SYMBOLS):
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/ne/news/{SYMBOLS}"

    headers = {
        "X-RapidAPI-Key": "0aaf3818f1msh30f338f1870b6d4p13f57djsnd74124e0100a",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers)
    news_list=json.loads(response.text)
    # print(news_list)
    news_data=[]
    try:
        i=0
        while(True):
            news_data.append([news_list["item"][i]['title'],news_list["item"][i]['description']])
            i=i+1
    except Exception as e:
        print(e)

    if news_data!=None:
        return pd.DataFrame(news_data)
    else:
        return None

def sentiment_analysis(text):
    return classifier(text)


def avg_sentiment_score(results):
    sum=0
    for i in results:
        sum=sum+i

    avg=sum/len(results)

    return avg


def sentiment_score(news):
    results=[]
    print(news)
    for i in range(0,10):
        query=news[0][i]+" "+news[1][i]
        try:
            result=sentiment_analysis(query)
            # print(result[0])
            if(result[0]['label']=='positive'):
                results.append(result[0]['score'])
            elif(result[0]['label']=='neutral'):
                results.append(result[0]['score']/-4)
            elif(result[0]['label']=='negative'):
                results.append(result[0]['score']*-1)

        except Exception as e:
            print(e)
    print(results)
    final_score = avg_sentiment_score(results)

    return final_score



def current_price(SYMBOL):
    url=f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{SYMBOL}/financial-data"
    headers = {
        "X-RapidAPI-Key": "0aaf3818f1msh30f338f1870b6d4p13f57djsnd74124e0100a",
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers)

    resp=json.loads(response.text)
    return resp['financialData']['currentPrice']['fmt']



#-----------------------------------------------------------------------#


st.write("""
# Stonks :money_with_wings:
*Your friendly neighbourhood Stock Buddy* :dizzy:
 
""")

enterTicker = st.text_input('Enter stock name: ', 'Google')



tickerSymbol=getTicker(enterTicker)

tickerData=yf.Ticker(tickerSymbol)



df=return_news(tickerSymbol)

if(df.empty):
    st.write(f"""
        ## Stocks not found!
    """)
else:
    currentPrice = current_price(tickerSymbol)

    tickerDf = tickerData.history(period='1d', start='2013-5-31', end='2023-1-10')
    st.line_chart(tickerDf.Close)

    d = tickerDf.Close.values.reshape([tickerDf.Close.values.shape[0], 1])
    X, y = create_dataset(d, 100)
    X = np.expand_dims(X, axis=-1)

    close_model = tf.keras.models.load_model('close.h5')
    close_pred = close_model.predict(X).mean() * 100

    high_model=tf.keras.models.load_model('high.h5')
    high_pred = close_model.predict(X).mean() * 100

    low_model=tf.keras.models.load_model('low.h5')
    low_pred = low_model.predict(X).mean() * 100

    col1,col2=st.columns(2)

    with col1:
        st.write(f"""
            CURRENT PRICE :dollar: : **{currentPrice}**
        """)
        st.write(f"""
            HIGH :point_up: : **{round(high_pred,2)}** 
        """)
    with col2:
        st.write(f"""
            PREDICTED PRICE :euro: **{round(close_pred,3)}**:
        """)
        st.write(f"""
        LOW :point_down: : **{round(low_pred,2)}**
        """)



    final_score = sentiment_score(df)
    print(final_score)
    if final_score > 0.4:
        st.write("""
            #### Analysing recent news articles...⏳
        """)
        st.write(f"""
            Based on the news articles, we give it a score of **{round(final_score, 4) * 10}**. 
        """)
        st.write("""
            (Range: -10 to 10)
        """)
        st.write("""
            ### *We would recommend you to invest in this stock!*        
        """)
    else:
        st.write("""
            #### Analysing recent news articles...⏳
        """)
        st.write(f"""
            Based on the news articles, we give it a score of **{round(final_score, 4)*10}** 
        """)
        st.write("""
            (Range: -10 to 10)
        """)
        st.write("""
            ### *Please don't buy these stocks!!*
        """)

    with st.sidebar:
        st.write("""
        # NEWS :newspaper:
        """)

        st.write(f"""
            #### -> {df[0][0]}
        """)

        st.write(f"""
            #### -> {df[0][1]}
            """)

        st.write(f"""
            #### -> {df[0][2]}
                """)

        st.write(f"""
            #### -> {df[0][3]}
                """)

        st.write(f"""
            #### -> {df[0][4]}
                """)

