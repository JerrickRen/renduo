# Report

# Summary

The presented dataset investigates the correlation between sentiment scores and return values in the S&P500 index within a time span ranging from 2 to 10 days. It comprises sentiment scores extracted from textual data sources and daily returns for each stock in the index. Analyzing this dataset can provide us with valuable insights into how sentiment affects stock returns across different time frames. The dataset is organized in a way that facilitates advanced analysis and modeling, making it a valuable asset for researchers and analysts interested in exploring the interplay between sentiment and financial markets.


# Data section

# 2.How are the return variables built and modified?-Mechanical description.

1.The 'date' and 'Filing Date' columns in the 'data1' dataframe are converted to datetime objects using the pd.to_datetime() function.

2.The 'Count' column is created in the 'data1' dataframe by subtracting the 'Filing Date' from the 'date' for each row and converting it to the number of days. If 'date' is before 'Filing Date', then '-2' is assigned to 'Count' for that row.

3.The 'filt' dataframe is created by filtering the 'data1' dataframe to only include rows where the 'Count' column is between 0 and 2 (inclusive).

4.The 'cumulative_return1' variable is created by grouping the 'filt' dataframe by 'Symbol' and calculating the cumulative return for each group. The cumulative return is calculated as the product of (1 + return) for all rows in the group, minus 1. This is done using the np.prod() function.

5.The 'filt1' dataframe is created by filtering the 'data1' dataframe to only include rows where the 'Count' column is between 3 and 10 (inclusive).

6.The 'cumulative_return2' variable is created by grouping the 'filt1' dataframe by 'Symbol' and calculating the cumulative return for each group, following the same method as in step 4.

7.The 'CR_t7' and 'CR_t2' dataframes are created by converting the 'cumulative_return1' and 'cumulative_return2' variables to dataframes and renaming the 'ret' column to 'ret_t7' and 'ret_t2', respectively.

8.The 'cumulative_returns' dataframe is created by merging the 'CR_t7' and 'CR_t2' dataframes on 'Symbol' using the pd.merge() function with 'how' set to 'right', which keeps all the rows from 'CR_t7' and only the matching rows from 'CR_t2'. The resulting dataframe is also validated to ensure that it is a one-to-one match on 'Symbol'. The 'indicator' argument is set to True to add a column that shows whether each row came from both dataframes ('both'), only the left dataframe ('left_only'), or only the right dataframe ('right_only').

9.The 'cumulative_returns' dataframe is printed to show the final results.

code:
# Convert date and filing_date columns to datetime objects
data1['date'] = pd.to_datetime(data1['date'])
data1['Filing Date'] = pd.to_datetime(data1['Filing Date'])

# Calculate days since filing for each row
data1['Count'] = data1.apply(lambda row: (row['date'] - row['Filing Date']).days if row['date'] >= row['Filing Date'] else -2, axis=1)

# Filling date is between 0 and 2
filt = data1[(data1['Count'] >= 0) & (data1['Count'] <= 2)]

# Group by ticker and calculate cumulative return with previous
cumulative_return1 = filt.groupby('Symbol')['ret'].apply(lambda x: np.prod(1+x)-1)

print(cumulative_return1)

# Filling date is between 3 and 10
filt1 = data1[(data1['Count'] >= 3) & (data1['Count'] <= 10)]

# Group by ticker and calculate cumulative return with previous
cumulative_return2 = filt1.groupby('Symbol')['ret'].apply(lambda x: np.prod(1+x)-1)

print(cumulative_return2)

# Merge the two datas
CR_t7 = pd.DataFrame(cumulative_return1).rename(columns={'ret':'ret_t7'})
CR_t2 = pd.DataFrame(cumulative_return2).rename(columns={'ret':'ret_t2'})
cumulative_returns = CR_t2.merge(CR_t7, how='right', on='Symbol', indicator=True, validate='1:1')

print(cumulative_returns)

# 3.How are the sentiment variables are built and modified?-Mechanical description.

The code defines regular expressions for different sentiment variables related to corporate earnings, investment opportunities, and economic indicators. Each sentiment variable has a positive and negative version. The regular expressions are created by combining a list of relevant words using the "join" method and enclosing them in parentheses. These regular expressions can then be used to search and count occurrences of sentiment-related words in textual data, such as news articles or social media posts.





```python
#corporate_earnings

#1.positive
corporate_earnings_pos = ['(profitable|successful|strong|growth|increase|revenue|improve|expansion|positive|gain|healthy|rise)']

#2.negative
corporate_earnings_neg = ['(missed|loss|decline|downgrade|disappointing|weak||underperform|negative|struggle|fall|slump|warning)']

#investment_opportunities

#1.positive
investment_opportunities_positive=['(profit|growth|opportunity|return|expansion|innovation|success|diversification|potential|value|market|liquidity)']

#2.negative
investment_opportunities_negative =['(loss|risk|decline|failure|uncertainty|instability|downfall|downturn|volatility|crash|bankruptcy|default)']


#Economic indicators 

#1.Positive
economic_indicators_positive =['(growth|employment|surplus|stimulus|GDP|job|confidence|investment|productivity|prosperity|recovery)']

#2.Negative
economic_indicators_negative = ['(recession|inflation|unemployment|deficit|debt|contraction|austerity|decline|slowdown|stagnation|inequality|bankruptcy)']
```

# 4.Why did you choose the three topics you did for the “contextual sentiment” measures?

Corporate earnings is an important factor to consider when calculating sentiment scores because it is a key indicator of a company's financial health and overall performance. Positive earnings reports typically indicate that a company is profitable and may be a good investment opportunity, while negative earnings reports suggest that a company is struggling and may not be a wise investment choice. By analyzing sentiment around corporate earnings, investors and analysts can gain valuable insights into market trends and make more informed investment decisions.

Corporate investment opportunities are important to calculate sentimental score because they can provide insights into the potential profitability and growth prospects of a company. Positive sentiment in this area can indicate that investors may see opportunities for increased profits and returns, while negative sentiment may suggest that there are significant risks or challenges to investing in a particular company or sector. 

Economic indicators are important to calculate the sentimental score because they provide insight into the overall health and direction of the economy. Positive economic indicators such as growth, employment, and investment can indicate a thriving economy, while negative indicators such as recession, inflation, and unemployment can suggest economic trouble.



# 5.Show and discuss summary stats of your final analysis sample

![ ](1679681587704.jpg)

1.Mean and Standard Deviation:

The mean and standard deviation of each variable (IO_negative, EI_positive, EI_negative, ret_t2, ret_t7) can provide a sense of the central tendency and spread of the data.
For instance, the mean value of ret_t2 is negative (-0.009370) which suggests that on average, the returns of the stocks after two days are lower than their initial price.

2.Quartiles:

The quartile information (25%, 50%, and 75%) can give us an idea of the distribution of the data and identify potential outliers.
For example, the 25% quartile of ret_t2 is -0.048807, which means that 25% of the data lies below this value.

# 6.Do your “contextual sentiment” measures pass some basic smell tests?

To determine if the "contextual sentiment" measures pass some basic smell tests, we need to consider if the values make sense and align with our expectations.

LM_positive and LM_negative: These measures represent the positive and negative sentiment extracted from the financial news using the LM algorithm. The mean value for LM_positive is 0.005191 and for LM_negative is 0.016099, which makes sense as we would expect there to be more negative sentiment in financial news.

CE_positive and CE_negative: These measures represent the positive and negative sentiment extracted from the financial news using the content analysis approach. The mean value for CE_positive is 0.003689 and for CE_negative is 2.000195, which seems unusual as the negative sentiment score is much higher than the positive sentiment score. This may indicate that the content analysis approach needs further investigation.

IO_positive and IO_negative: These measures represent the positive and negative sentiment extracted from insider trading activity. The mean value for IO_positive is 0.006144 and for IO_negative is 0.002879, which aligns with our expectations that insider trading activity is more likely to indicate positive sentiment.

EI_positive and EI_negative: These measures represent the positive and negative sentiment extracted from external investor sentiment. The mean value for EI_positive is 0.001628 and for EI_negative is 0.001423, which are relatively low, but this could be due to the nature of external investor sentiment being more reactive to market changes.


# Results





# 1.Make a table with the correlation of each (10) sentiment measure against both (2) return measures. 

# 2.Include a scatterplot (or similar) of each sentiment measure against both return measures.

![ ](1.png)
![ ](2.png)
![ ](3.png)
![ ](4.png)
![ ](5.png)
![ ](6.png)
![ ](7.png)
![ ](8.png)
![ ](9.png)
![ ](10.png)


# Four discussion topics:

1.# Compare / contrast the relationship between the returns variable and the two “LM Sentiment” variables (positive and negative) with the relationship between the returns variable and the two “ML Sentiment” variables (positive and negative). Focus on the patterns of the signs of the relationships and the magnitudes.

Positive:
The scatter plot reveals that the majority of data points for day t to day t+2 fall within the sentiment score range of 0.02 to 0.08 and return range of -0.2 to 0.2. This pattern continues for day t+3 to day t+10. However, the red data points have more outliers than the blue data points.

Negative:
Analysis of the scatter plot shows that the majority of data points for day t to day t+2 are concentrated within the sentiment score range of 0.01 to 0.25 and return range of -0.2 to 0.2. This trend persists for day t+3 to day t+10. However, the red data points have more outliers compared to the blue data points.


2.If your comparison/contrast conflicts with Table 3 of the Garcia, Hu, and Rohrer paper (ML_JFE.pdf, in the repo), discuss and brainstorm possible reasons why you think the results may differ. If your patterns agree, discuss why you think they bothered to include so many more firms and years and additional controls in their study?

There are various reasons why our results may differ from the patterns observed in Table 3 of the Garcia, Hu, and Rohrer paper (ML_JFE.pdf). These differences could be attributed to variations in sample size, control variable selection, data sources, time period studied, and modeling techniques. Alternatively, if our findings are consistent with those in Table 3, it is possible that Garcia, Hu, and Rohrer included additional firms, years, and controls to ensure the robustness of their model. This approach could have helped them account for any confounding variables that could have affected the relationship between the independent and dependent variables. By including more controls, they might have been able to accurately estimate the effect of the variables of interest and reduce any potential bias in their results. Furthermore, having a larger sample size might have improved the generalizability of their findings.

3.Discuss your 3 “contextual” sentiment measures. Do they have a relationship with returns that looks “different enough” from zero to investigate further? If so, make an economic argument for why sentiment in that context can be value relevant.

First, corporate earnings sentiment has a positive relationship with returns, indicating that a positive sentiment about a company's earnings is associated with higher returns. This makes economic sense because investors often consider a company's earnings when making investment decisions. Positive earnings sentiment may suggest that a company is performing well and is expected to generate higher profits, which can lead to increased investor confidence and demand for the company's stock, driving up its price.

Second, corporate investment opportunities sentiment also has a positive relationship with returns, suggesting that positive sentiment about a company's investment opportunities is associated with higher returns. This can be explained by the fact that investors are attracted to companies with strong growth potential, as such companies are likely to generate higher profits in the future. Positive sentiment about a company's investment opportunities can signal that the company is expected to grow and succeed in the future, making its stock more attractive to investors and driving up its price.

Third, economic indicators sentiment has a negative relationship with returns, indicating that positive sentiment about economic indicators is associated with lower returns. This may seem counterintuitive at first, but it can be explained by the fact that positive economic sentiment often leads to higher interest rates, which can reduce demand for stocks as investors shift their investments to fixed-income securities. Additionally, high economic sentiment can lead to inflationary pressures, which can negatively impact the economy and decrease stock returns.
