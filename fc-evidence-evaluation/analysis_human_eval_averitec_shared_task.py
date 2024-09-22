import pandas as pd
import matplotlib.pyplot as plt
import os

# File path
_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/averitec_shared_task/averitec_shared_task_eval_scores.xlsx"
_PATH_DIRECTORY = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/averitec_shared_task/plots"
# 1. Load Excel file into dataframe
df = pd.read_excel(_PATH)

# 2. Plot and save distribution of values in various columns as histograms
columns_to_plot = ['semantic_coverage', 'coherence', 'redundancy', 'consistency', 'relevance']

# 3. Column 'predicted evidence' contains text, split by space and calculate length
df['text_length'] = df['predicted evidence'].apply(lambda x: len(str(x).split()))

# Plot and save resulting text length as a histogram
plt.figure(figsize=(8, 6))
plt.hist(df['text_length'], bins=30, edgecolor='black')
plt.title('Distribution of Text Length in Predicted Evidence')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.savefig(os.path.join(_PATH_DIRECTORY, 'distribution_text_length.png'))
plt.show()

# 4. Plot and save the correlation between text length and 'hmeteor' values
plt.figure(figsize=(8, 6))
plt.scatter(df['text_length'], df['hmeteor'])
plt.title('Text Length vs. Hmeteor')
plt.xlabel('Text Length')
plt.ylabel('Hmeteor')
plt.savefig(os.path.join(_PATH_DIRECTORY, 'text_length_vs_hmeteor.png'))
plt.show()
