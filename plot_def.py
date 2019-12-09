import matplotlib.pyplot as plt
import os

labels = ['airplane', 'automobile', 'bird', 'cat',  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
FILE_PATH = 'static/prediction.png'

def remove_if_exists():
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)


def generate_pred_graph(label):
    remove_if_exists()

    plt.figure(figsize=(10, 2))

    values = [0 for c in range(10)]
    values[labels.index(label)] = 1
    
    print(values)
    plt.bar(labels, values)
    plt.suptitle('Result')
    plt.savefig(FILE_PATH)
    plt.close()