import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../dataset/training.csv")
    corr = df.corr(numeric_only=True, method="pearson")
    mask = np.triu(np.ones_like(corr))
    sns.heatmap(corr, annot=True, mask=mask)
    plt.show()
