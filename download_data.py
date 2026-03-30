#!/usr/bin/env python3
"""
Script to download the diabetic patient dataset.

This script downloads the Diabetes 130-Hospitals dataset using the Fairlearn library.
"""

def download_data():
    """Download the diabetes hospital dataset."""
    try:
        from fairlearn.datasets import fetch_diabetes_hospital
        
        print("Downloading diabetes hospital dataset...")
        df = fetch_diabetes_hospital(as_frame=True)
        
        print(f"Dataset shape: {df.frame.shape}")
        print(f"\nColumn names:")
        print(df.frame.columns.tolist())
        
        df.frame.to_csv('data/diabetic_data.csv', index=False)
        print(f"\nData saved to: data/diabetic_data.csv")
        
    except ImportError:
        print("Fairlearn not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'fairlearn'])
        download_data()


if __name__ == "__main__":
    download_data()
