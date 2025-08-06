import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("finalized.csv")

# VISUAL ONE
# displaying the watched and not watched
# sns.countplot(x='watched', data=df)
# plt.title("Watched vs Unwatched Movies")
# plt.xlabel("Watched (0 = No, 1 = Yes)")
# plt.ylabel("Quantity")
# plt.show()


# VISUAL TWO
##### Old ones, no labels for meadian #####
# sns.boxplot(x='watched', y='rating', data=df)
# plt.title("Movie Rating by Watched Status")
# plt.show()
# sns.boxplot(x='watched', y='duration', data=df)
# plt.title("Duration by Watched Status")
# plt.show()
##### Curent ones, with labels for median #####
# plt.figure(figsize=(6, 4))  # this represents the window of the results
# ax1 = sns.boxplot(x='watched', y='rating', data=df)
# plt.title("Movie Rating by Watched Status")
# styling the median value
# medians = df.groupby('watched')['rating'].median()
# for i, median in enumerate(medians):
#     ax1.text(i, median, f'{median:.2f}', ha='center', va='center',
#              fontweight='bold', color='green', bbox=dict(facecolor='white'))
# plt.show()
# Second plot: Duration by Watched
# plt.figure(figsize=(6, 4))
# ax2 = sns.boxplot(x='watched', y='duration', data=df)
# plt.title("Duration by Watched Status")

# # styling the median value
# medians = df.groupby('watched')['duration'].median()
# for i, median in enumerate(medians):
#     ax2.text(i, median, f'{median:.0f}', ha='center', va='center',
#              fontweight='bold', color='green', bbox=dict(facecolor='white'))
# plt.show()


# VISUAL THREE
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()






