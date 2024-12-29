# Define numerical & categorical columns
def print_feature_types(df):
    numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
    categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

    # Print columns
    print(f'We have {len(numeric_features)} numerical features : {numeric_features}')
    print(f'\nWe have {len(categorical_features)} categorical features : {categorical_features}')


# categorical columns
def print_categories(df):
    columns = df.select_dtypes(include=['object']).columns
    for col in columns:
        print(f"Categories in \033[1m'{col}'\033[0m there are \033[1m{df[col].nunique()}\033[0m categories: \033[1m{df[col].unique()}\033[0m")

# tenure segment functions
def map_tenure_segment(tenure):
    if tenure <= 6:
        return '1. 0 - 6 months'
    elif 7 <= tenure <= 12:
        return '2. 7 - 12 months'
    elif 13 <= tenure <= 18:
        return '3. 13 - 18 months'
    elif 19 <= tenure <= 24:
        return '4. 19 - 24 months'
    else:
        return '5. > 24 months'
    

# Average, Median and Standard deviation of Tenure, Monthly Charges, and Total Charges
def calculate_key_metrics(df, numeric_columns=None, handle_missing='drop'):
    """
    Calculate key metrics for specified numeric columns in the DataFrame.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - numeric_columns (list): List of numeric column names to summarize. If None, all numeric columns are considered.
    - handle_missing (str): 'drop' to drop missing values, 'fill' to fill them with the column mean, or 'ignore'.
    
    Returns:
    - pandas.DataFrame: DataFrame containing the calculated key metrics with no index changes.
    """
    if numeric_columns is None:
        # Automatically select numeric columns if none are specified
        numeric_columns = df.select_dtypes(include='number').columns.tolist()

    # Handle missing values based on the chosen method
    if handle_missing == 'drop':
        df = df.dropna(subset=numeric_columns)
    elif handle_missing == 'fill':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Initialize the key metrics dictionary
    key_metrics = {}

    # Calculate the key metrics for each numeric column dynamically
    for col in numeric_columns:
        if col in df.columns:
            key_metrics.update({
                f"Average {col}": df[col].mean(),
                f"Median {col}": df[col].median(),
                f"Std Dev {col}": df[col].std()
            })

    # Convert to a DataFrame and round the values for display
    key_metrics_df = pd.DataFrame(list(key_metrics.items()), columns=["Metric", "Value"])
    key_metrics_df["Value"] = key_metrics_df["Value"].round(2)  # Round all values to 2 decimal places

    return key_metrics_df


# Calculate Active vs. Churned distribution
def calculate_status_distribution(df, column_name):
    """
    Calculate the count and percentage distribution of values in a binary categorical column.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to analyze (e.g., 'Churn').
    
    Returns:
    - pandas.DataFrame: DataFrame with counts and percentages for each category.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    # Total customers
    total_customers = df.shape[0]

    # Count values and calculate percentages
    value_counts = df[column_name].value_counts()
    percentages = round((value_counts / total_customers) * 100, 2)
    
    # Create a DataFrame for the results
    distribution_data = {
        f"{column_name}": value_counts.index.tolist(),
        "Count": value_counts.values.tolist(),
        "Percentage": percentages.values.tolist()
    }
    
    distribution_df = pd.DataFrame(distribution_data)
    
    return distribution_df


