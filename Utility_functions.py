import matplotlib.pyplot as plt

def plot_figure(X,Y,xlabel,ylabel,title):
    # plot
    plt.figure()
    plt.plot(X,Y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
