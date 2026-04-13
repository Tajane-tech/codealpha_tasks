import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in India.csv")

df.columns = df.columns.str.strip()

df['Date'] = df['Date'].str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df.dropna(inplace=True)

df['Year'] = df['Year'].astype('Int64')
df['Month'] = df['Month'].astype('Int64')

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())

plt.figure(figsize=(10,5))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df)
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show()

year_avg = df.groupby('Year')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(8,5))
year_avg.plot(kind='bar')
plt.title("Year-wise Average Unemployment Rate")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

before_covid = df[df['Year'] < 2020]['Estimated Unemployment Rate (%)'].mean()
after_covid = df[df['Year'] >= 2020]['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Before COVID:", before_covid)
print("Average Unemployment After COVID:", after_covid)

region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
region_avg.plot(kind='bar')
plt.title("Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.show()

month_avg = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(8,5))
month_avg.plot(kind='line', marker='o')
plt.title("Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.show()