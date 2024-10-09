import joblib
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

custom_cmap = "viridis"  # "coolwarm" #"YlGnBu"

#def R_correction(data_features):
#    dcorr = data_features.corr(method='pearson').dropna(how='all', axis=0).dropna(how='all', axis=1)
#
#    mask = np.triu(np.ones_like(dcorr))
#
#    plt.figure(figsize=(15, 12), dpi=100)
#
#    ax = sns.heatmap(data=dcorr,
#                # vmax=1.0, vmin=0.5,
#                # annot=True,
#                # linewidths=0.4,
#                # cmap="Spectral",
#                # cbar_kws={"orientation":"horizontal","location":"bottom"},
#                cmap=custom_cmap,
#                mask=mask)
#
#    plt.tick_params(axis='both', labelsize=20)
#    plt.tight_layout()
#    cbar = ax.collections[0].colorbar
#    cbar.ax.tick_params(labelsize=20)  # 设置色度条刻度标签的字体大小
#    plt.savefig("features_correction.png")
#    return dcorr
    
def R_correction(data_features):

    # 计算相关性矩阵并去除全空行列
    dcorr = data_features.corr(method='pearson').dropna(how='all', axis=0).dropna(how='all', axis=1)

    # 创建上三角遮罩
    mask = np.triu(np.ones_like(dcorr))

    # 动态调整图表大小
    plt.figure(figsize=(15, 12), dpi=500)

    # 创建热图
    heatmap = sns.heatmap(data=dcorr,
                          vmax=0.8, 
                          vmin=-0.8,
                          cmap=custom_cmap,
                          mask=mask)

    # 修改色度条的标签字号
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=22)
    colorbar.ax.yaxis.label.set_size(22)

    # 设置标签
    y_label = data_features.columns.to_list()
    x_label = data_features.columns.to_list()
    heatmap.set_yticklabels(y_label, rotation=0, size=28)
    heatmap.set_xticklabels(x_label, rotation=-45, ha='left', rotation_mode='anchor',  size=28)  # 45度旋转，并右对齐

    # 调整标签的布局，确保不重叠
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor', size=26)
    plt.yticks(size=26)

    # 自动调整布局以避免重叠
    plt.tight_layout()
    plt.savefig("features_correction.png")
    plt.show()
    
    return dcorr

def remove_high_R(X, threshold=0.9):  # X is a pandas DataFrame that only includes features
    delet_index = []
    delet_dic = {}
    dcorr = X.corr(method='pearson')
    for columns in dcorr:
        # print("columns:", columns)
        if columns in delet_index:
            continue
        else:
            dcorr_i_sort = abs(dcorr.loc[columns]).sort_values(ascending=False)
            columns_in = dcorr_i_sort.index
            columns_list = list(columns_in)
            columns_list.remove(columns)
            dcorr_i_sort_pd = pd.DataFrame(dcorr_i_sort)
            index_j = []
            for inr in columns_list:
                dcorr_ij = dcorr_i_sort.loc[inr]
                if dcorr_ij >= threshold:
                    index_j.append(inr)
                else:
                    break
            delet_index.extend(index_j)
    left_index = [i for i in X.columns if i not in delet_index]

    print("total features:", len(X.columns))
    print("deleted features:", len(list(set(delet_index))))
    print("left features:", len(left_index))
    return delet_index, left_index

###########################################################

filename = "features_training.xlsx"
data_targets = pd.read_excel(filename)
data_features = data_targets.loc[:,"mol_weight":]

delet_index, left_index = remove_high_R(data_features,threshold=0.9)

# left_index: ['mol_weight', 'mol_volume', 'mol_polar_area', 'mol_surface_area', 'logP'
            # , 'num_rotatable_bonds', 'ring_count', 'num_h_donors', 'num_h_acceptors'
            # , 'surface_area', 'estate_indices_std', 'estate_indices_sum'
            # , 'estate_indices_mean', 'chiral_centers', 'radius_of_gyration'
            # , 'chi1v', 'counter_electronegativity', 'diameter', 'bond_lengths_mean'
            # , 'bond_angles_mean', 'torsions_std']

used_ = ['mol_weight', 'mol_volume', 'mol_polar_area', 'mol_surface_area', 'logP'
            , 'num_rotatable_bonds', 'ring_count', 'num_h_donors', 'num_h_acceptors'
            , 'surface_area', 'estate_indices_std', 'estate_indices_sum'
            , 'estate_indices_mean', 'chiral_centers', 'radius_of_gyration'
            , 'chi1v', 'counter_electronegativity', 'diameter', 'bond_lengths_mean'
            , 'bond_angles_mean', 'torsions_std']

# ['mol_volume', 'mol_polar_area', 'logP', 'num_h_donors', 'num_h_acceptors', 'estate_indices_std', 'estate_indices_mean', 'chi1v', 'bond_lengths_mean']
# recolumns_ = [r"$V_{m}$","$A_{polar}$", "$logP$", "$N_{H\_donor}$", "$N_{H\_acceptor}$","$E_{state,std}$","$E_{state,mean}$","$Chi1V$","$L_{mean}$"]
recolumns_ = [r"$M_{w}$", "$V_{m}$", "$A_{polar}$", "$A_{m}$", "$logP$"
,"$N_{rot}$", "$N_{ring}$", "$N_{H\_donor}$", "$N_{H\_acceptor}$"
,"$A$","$E_{state,std}$","$E_{state,sum}$"
,"$E_{state,mean}$","$N_{chiral}$","$R_{g}$"
,"$Chi1V$", "$N_{χ>2.55}$","$d$","$L_{mean}$"
,"$θ_{mean}$", "$Torsion_{std}$"]

input_features = data_features.drop(delet_index,axis=1)
input_features = input_features[used_]
input_features.columns = recolumns_
R_correction(input_features)

with open("features.txt", "w") as file:
    file.write(f"delet_index: {delet_index}\n")
    file.write(f"left_index: {left_index}\n")

