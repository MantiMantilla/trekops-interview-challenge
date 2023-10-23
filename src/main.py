# # Trekops Data Analytics Challenge
#
# This notebook is my take on the Trekops Data Analytics Challenge for the Data Analyst job application.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression

# ## 1. Importing and Cleaning the Data

# Read xlsx file using pandas
payments_df = pd.read_excel("../data/Recruiting Task Dataset.xlsx", header=0)
payments_df.head()

# Identify missing values
payments_df.isnull().sum(axis=0)

# There seem to be no missing values except for two unidentified Issuing Bank registers.
# I attempted to correct this by viewing what was common between these to rergisters
# or how they relate to other registers, but found no discernible pattern, other than
# both being rejected transaction attempts.
# I correct this by simply adding a new category of "Other" that will match only
# these two registers.

payments_df["Issuing Bank"].fillna("Other", inplace=True)
payments_df.isnull().sum(axis=0)

# Identify dtypes and correct mislabeled types (Amount columns should be float).
print(payments_df.dtypes)
payments_df["Amount"].replace(r"[\$,]", "", regex=True).astype(float)
payments_df["Amount"].head()

# ## 2. Questions

# ### i. What is the dataset's approval rate by quarter?

# Introduce a `Quarter` column to group with Pandas. Resembles SQL's GROUP BY directive.
payments_df["Quarter"] = pd.PeriodIndex(payments_df["Attempt Timestamp"], freq="Q")
grouped_df = payments_df.groupby("Quarter").agg({"Appr?": ["sum", "count"]})
grouped_df.columns = grouped_df.columns.droplevel(0)
rates = grouped_df["sum"] / grouped_df["count"]
print(rates)

sns.scatterplot(x=rates.index.astype("datetime64[ns]"), y=rates)
plt.show()

# ### ii. How many customers attempted a deposit of $50 in Sept 2021?

# Filter by date and payment amount, then count unique CustomerIDs.
is_month = payments_df["Attempt Timestamp"].dt.month == 9
is_year = payments_df["Attempt Timestamp"].dt.year == 2021
is_amount = payments_df["Amount"] == 50
users = payments_df[is_month & is_year & is_amount]["CustomerID"]
distinct_attempts = len(users.unique())
total_attempts = len(users)

print(
    f"There were {distinct_attempts} distinct clients, among {total_attempts} total attempts."
)

# ### iii.How much did the group identified in QUESTION 2 successfully deposit during the month?

# Filter again, also by transaction approval, don't filter $50 transactions and distinct customers.
is_approved = payments_df["Appr?"] == 1
amounts = payments_df[is_month & is_year & is_approved]["Amount"]
print(f"The total transacted amount is ${amounts.sum() :,.2f}.")

# ### iv. Of the 10 issuing banks with the most deposit attempts between $150.00 and $999.99 in 2021, which had the highest approval rate for the attempts of that deposit amount?

# Filter by deposit amount and year. Group by bank, count attempts.
is_in_range = (payments_df["Amount"] >= 150) & (payments_df["Amount"] < 1000)
is_year = payments_df["Attempt Timestamp"].dt.year == 2021
top_banks = list(
    payments_df[is_in_range & is_year]
    .groupby("Issuing Bank")["Issuing Bank"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# Filter by amount, year, and bank. Calculate approval rate.
is_top_bank = payments_df["Issuing Bank"].isin(top_banks)
grouped_df = (
    payments_df[is_in_range & is_year & is_top_bank]
    .groupby("Issuing Bank")
    .agg({"Appr?": ["sum", "count"]})
)
grouped_df.columns = grouped_df.columns.droplevel(0)
rates = grouped_df["sum"] / grouped_df["count"]
print(rates)
print(
    f"The top issuing bank with highest approval rate between $150.00 and $999.99 from 2021 is {rates.idxmax()}"
)

# ### v. Without performing any analysis, which two parameters would you suspect of causing the successive quarterly decrease in approval rate? Why?

# I would expect the most influential factor would be the Deposit Amount. As the year goes on, people might have different spending habits and would most likely spend more later in the year and settle into a routine. Greater transaction amounts (both inside and outside the platform) are likely inversely correlated with approval, and so attempted transactions later in the year would be most likely to fail. Insufficient funds and accelerated spending would moslikely result in more rejected transactions.

# I would also expect the Website to which customers submit their payment to be of great influence. As the year goes, users might migrate from one platform to another and as such would inevitably face some friction in performing their transactions. Not only would a bank likely flag these changes as fraud, the customers themselves might make mistakes in registering their information to a new website. Beyond that, a few platforms are likely to move the most volume by transaction amount and transaction count, and so these are likely to have a different approval rate with respect to other websites.

# ### vi. Identify and describe two main causal factors of the decline in approval rates seen in Q3 2021 vs Q4 2020?

# Without getting into predictive or inferential models, I identify the most influential factors according to different importance metrics.

# Plot approval with respect to each other feature.
sns.pairplot(payments_df)
plt.show()


# Mutual information:


# ### vii. Choose one of the main factors identified in QUESTION 6. How much of the approval rate decline seen in Q3 2021 vs Q4 2020 is explained by this factor?

# ### viii. If you had more time, which other analyses would you like to perform on this dataset to identify additional causal factors to those identified in QUESTION 6.
