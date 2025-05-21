import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def pr(error):
    
    probs = 1e5/(1e5 + error)

    return probs

def prob_for_all(unknown_amino, amino_dic):
    error = []
    keys = []
    
    norm = 40000/np.sum(unknown_amino)
    for key, value in amino_dic.items():
        
        error.append(np.sum((unknown_amino - value/norm)**2))
        keys.append(key)
    
    probability = pr(np.array(error))

    return probability

 
def main():


    probs = np.zeros((17,17))
    acids = []
    loaded = np.load("data_amino_acids_dictionary.npz")
    i = 0
    for amino, _ in loaded.items():
        
        filename = "data_csv_filer_unknown/" + amino + "_unknown_4000.csv"
        data = np.loadtxt(filename, delimiter=",")
        pr = prob_for_all(data, loaded)
        probs[i] = pr
        acids.append(amino)
        i +=1
        print(f"{i} aminosyra klar")
    
    probs = probs*100
    
    norm = mcolors.PowerNorm(gamma= 2, vmin=0, vmax=100)

    df = pd.DataFrame(probs, index=acids, columns=acids)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='RdPu',norm = norm , fmt=".2f") 
    plt.title("Probabilities for matching amino acid in %")
    plt.xlabel("Reference")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()