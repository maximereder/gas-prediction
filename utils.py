import numpy as np
import matplotlib.pyplot as plt


def df_to_X_y2(df, window_size=6):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
      row = [r for r in df_as_np[i:i+window_size]]
      X.append(row)
      label = df_as_np[i+window_size][0]
      y.append(label)
    return np.array(X), np.array(y)


def plot(df, title, x_label, y_label):
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(df.columns.values, shadow=True)
    plt.savefig(title + '.png')