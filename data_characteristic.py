import src

def print_section(title):
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)

def overview_data(df, name):
    print_section(f"Overview of {name}")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe(include="all").transpose().head(10))  

def check_missing_values(df, name):
    missing_values = df.isnull().sum()
    missing = missing_values[missing_values > 0]
    if not missing.empty:
        print(missing)
        print("\nPercentage of missing values:")
        print((missing / len(df) * 100).round(2))
    else:
        print("✅ No missing values")
    print("Total missing values:", missing_values.sum())

def check_duplicates(df, name):
    duplicate_rows = df[df.duplicated()]
    if len(duplicate_rows) > 0:
        print("Number of duplicate rows:", len(duplicate_rows))
    else:
        print("✅ No duplicate rows found")

def data_distribution(df, column, name):
    print("Value counts:")
    print(df[column].value_counts().head(10)) 
    print("\nPercentage distribution:")
    print((df[column].value_counts(normalize=True).head(10) * 100).round(2))

if __name__ == "__main__":
    # df1
    overview_data(src.df1, "df1")
    check_missing_values(src.df1, "df1")
    check_duplicates(src.df1, "df1")
    data_distribution(src.df1, "Country", "df1")

    # df2
    overview_data(src.df2, "df2")
    check_missing_values(src.df2, "df2")
    check_duplicates(src.df2, "df2")
    data_distribution(src.df2, "Product Category", "df2")

    # df_cust
    overview_data(src.df_cust, "df_cust")
    check_missing_values(src.df_cust, "df_cust")
    check_duplicates(src.df_cust, "df_cust")
    data_distribution(src.df_cust, "Gender", "df_cust")
    data_distribution(src.df_cust, "Age", "df_cust")
