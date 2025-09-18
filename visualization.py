import src

def plot_histogram(df, column, title="Histogram"):
    src.plt.figure(figsize=(8, 5))
    df[column].hist(bins=10000, edgecolor='black')
    src.plt.xlim(0, 300)  #
    src.plt.title(f"{title}: {column}")
    src.plt.xlabel(column)
    src.plt.ylabel("Frequency")
    src.plt.grid(False)
    src.plt.show()

def plot_bar_distribution(df, column, title="Bar Chart"):
    src.plt.figure(figsize=(10, 6))
    df[column].value_counts().head(10).plot(kind='bar', color='skyblue')
    src.plt.title(f"{title}: {column}")
    src.plt.xlabel(column)
    src.plt.ylabel("Count")
    src.plt.xticks(rotation=45)
    src.plt.show()

def plot_boxplot(df, column, by=None, title="Boxplot"):
    src.plt.figure(figsize=(8, 5))
    if by and by in df.columns:
        src.sns.boxplot(x=df[by], y=df[column])
        src.plt.title(f"{title}: {column} by {by}")
    else:
        src.sns.boxplot(y=df[column])
        src.plt.title(f"{title}: {column}")
    src.plt.show()

def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    src.plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)  # chỉ số dạng số
    src.sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    src.plt.title(title)
    src.plt.show()


if __name__ == "__main__":
    # df1 visualizations
    plot_histogram(src.df1, "Quantity", title="df1 Quantity Distribution")
    plot_bar_distribution(src.df1, "Country", title="df1 Country Distribution")
    plot_correlation_heatmap(src.df1, title="df1 Correlation Heatmap")

    # df2 visualizations
    plot_bar_distribution(src.df2, "Product Category", title="df2 Product Category Distribution")
    plot_correlation_heatmap(src.df2, title="df2 Correlation Heatmap")

    # df_cust visualizations
    plot_bar_distribution(src.df_cust, "Age", title="df_cust Age Distribution") 