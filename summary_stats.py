import pandas as pd
import os

def main():
    df = pd.read_csv('./dataset/Daily_activity_metrics.csv')
    
    print("=== DATA INFO ===")
    df.info()                           # rows, cols, dtypes, non-null counts
    
    print("\n=== SHAPE ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # 3) Missing values
    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum())
    
    # 4) Unique counts (good for categorical columns)
    print("\n=== UNIQUE VALUES PER COLUMN ===")
    print(df.nunique())
    
    # 5) Basic numeric summary
    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe().T)              # transpose for easier reading

    
if __name__ == "__main__":
    main()