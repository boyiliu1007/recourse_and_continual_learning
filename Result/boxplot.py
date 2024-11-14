import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import textwrap

def draw(all = False):
    if not all:
        # Step 1: Read the CSV file
        file_path = 'Result/topk_continual.csv'
        df = pd.read_csv(file_path)

        labels = []
        for column in df.columns:
            labels.append(column.replace('_', '_\n'))
        
        df.columns = labels
        # Step 2: Plot the Box Plot for all columns
        sns.boxplot(data=df)

        # Add title and labels
        plt.title('Failed to Recourse Box Plot')
        plt.xlabel('Columns')
        plt.ylabel('Fail to Recourse Percentage')

        # Show the plot
        plt.savefig('Result/boxplot.png')
        plt.show()
        
    
    else:
        csv_files = glob.glob('Result/*.csv')
        data = []
        labels = []

        # Step 2: Combine the 'a' Columns
        for file in csv_files:
            df = pd.read_csv(file)  # Read each CSV file
            for column in df.columns:
                labels.append(column.replace('_', '_\n'))
            data.append(df)  # Extract the 'a' column and add it to the list

        # Combine all 'a' columns into a single DataFrame
        combined_df = pd.concat(data, ignore_index=True)
        combined_df.columns = labels

        # Step 3: Plot the Combined Data
        sns.boxplot(data=combined_df)

        # Add title and labels
        plt.title('Box Plot')
        plt.xlabel('Experiments')
        plt.ylabel('Fail to Recourse Ratio')

        # Show the plot
        plt.savefig('Result/boxplot.png')
        plt.show()

draw()