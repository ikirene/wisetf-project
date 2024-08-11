# **MOTIVATION**

I have always believed that financial education is a powerful tool for anyone aiming to improve their quality of life with efficiency and intelligence. However, the field of finance is vast and can be overwhelming for someone new to investing. For this reason, I have focused this project on the study and analysis of ETFs.

Exchange Traded Funds (ETFs) are a type of fund traded on stock exchanges designed to replicate the performance of an index, sector, commodity, or a group of assets. ETFs have the following characteristics:

**Diversification**: ETFs allow investors to have diversified exposure to a group of assets with a single investment, reducing risk compared to buying individual stocks.

**Accessibility**: ETFs are traded on stock exchanges like stocks, making them easy to buy and sell during market hours.

**Low Costs**: Generally, ETFs have lower fees compared to traditional mutual funds due to their passive management.

**Transparency**: ETFs typically disclose their holdings daily, allowing investors to know exactly what they are investing in.

**Taxation**: In Spain, both the transfer and capital gains from ETFs are taxed at 21% - 28%, depending on the amount to be taxed.

**Trading**: Although not typically used for trading, ETFs offer great flexibility and immediacy when buying and selling.

The goal of this project is to provide an approach to the ETF market so that small investors can have more clarity when choosing among the thousands of available ETFs.

The results of our model can be summarized in two complementary phases (the techniques will be detailed later):

Medium to long-term prediction of an ETF’s closing price (Regression)
Portfolio optimizer of top ETFs based on investor preferences and predictions made

# **ORGANIZATION**

## **Data Sources**

The use of these sources is completely free.

**ETFdb**: To obtain the tickers of all ETFs classified by assets.

**yfinance**: To obtain the time series data.

## **Period**

We will work with data from 2018-01-01 to 2024-07-11.
The time series will have a period of 1 day ('1d').

## **Technical Indicators**

To make stock market predictions, these technical indicators are used in finance, many of which are Moving Averages. We could include more.

**RSI (Relative Strength Index):** Helps detect potential entry or exit points. It measures overbought or oversold conditions.

**EWMA 7, 50, 200**

**Price changes (1-5 days)**


