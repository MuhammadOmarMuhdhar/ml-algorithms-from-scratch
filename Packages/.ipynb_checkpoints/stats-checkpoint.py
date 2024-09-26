import numpy as np

def remove_outliers(df, column, lower_percentile=25, upper_percentile=75, iqr_multiplier=1.5):
    """
    Remove outliers from a DataFrame column based on a customizable IQR method.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to remove outliers from.
    - lower_percentile (float): The lower percentile to use (default is 25).
    - upper_percentile (float): The upper percentile to use (default is 75).
    - iqr_multiplier (float): The multiplier for the IQR to define outliers (default is 1.5).
    
    Returns:
    - DataFrame: A new DataFrame with outliers removed.
    """
    
    lower_bound = np.percentile(df[column], lower_percentile)
    upper_bound = np.percentile(df[column], upper_percentile)

    IQR = upper_bound - lower_bound
    
    lower_bound -= iqr_multiplier * IQR
    upper_bound += iqr_multiplier * IQR
    
    # Remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

