import matplotlib.pyplot as plt
import numpy as np
import pickle

def main():
    ###### edit here #####
    data_path = '/home/yoojinoh/winter_internship/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces/artifacts/src/rNetCLR/finetuning/results/nj_4_e__100_1/nj_4_e__100_graph_1.pickle'
    save_path = '/home/yoojinoh/winter_internship/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces/artifacts/src/rNetCLR/finetuning/results/nj_4_e__100_1/nj_4_e__100_graph_1.png'
    Title = 'noise injection intv:2.0 100(20)'
    #######################

    # 1. load data
    with open(data_path, 'rb') as f:
        results = pickle.load(f) 
    print(f"Successfully loaded pickle file {data_path.split('/')[-1][:-7]}")
    # 2. show data
    print(results.keys()) # dict_keys(['N', 'x', 'sup_y', 'inf_y', 'best_sup_y', 'best_inf_y'])

    # 3. Compare the best accuracy to choose the pair to draw
    best_sups = np.array(results['best_sup_y'])
    best_infs = np.array(results['best_inf_y'])

    best_idx = np.argmax(best_sups) ##
    print(f'best accuracy came from trial {best_idx + 1}(=index {best_idx})') # 4
    best_infs_idx = np.argmax(best_infs)

    N = results['N'][best_idx]
    x = np.array(results['x'][best_idx])
    sup_y = np.array(results['sup_y'][best_idx])
    inf_y = np.array(results['inf_y'][best_idx])
    # x = np.array(results['x'])[100 *(best_idx -1) : 100*best_idx] # 총 5세트, 각 100번 에포크 -> 0~100, 100~200, 200~300, 300~400
    # sup_y = np.array(results['sup_y'])[100 *(best_idx -1) : 100*best_idx]
    # inf_y = np.array(results['inf_y'])[100 *(best_idx -1) : 100*best_idx]

    print(f'Sup best acc {max(sup_y)} at {np.argmax(sup_y)}') # epoch starts from 0
    print(f'Inf best acc {max(inf_y)} at {np.argmax(inf_y)}') 
    print(f'acc of inf at epoch_sup_best is {inf_y[np.argmax(sup_y)]}')
    # sup = red, inf = blue
    plt.xlabel('Epoch')
    plt.ylabel('accuracy(%)') 

    plt.plot(x, sup_y, color = 'lightcoral', label = 'superior',marker='o')
    plt.plot(x, inf_y, color = 'cornflowerblue', label = 'inferior',marker='s')

    plt.legend()
    plt.title(f'Accuracy: size-{Title}-{N}') # N = 1,2,3,4,5
    plt.savefig(save_path)
    print('Saved the graph !')

if __name__ == "__main__":
    main()