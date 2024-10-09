from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors, EState, rdMolTransforms
from rdkit.Chem.rdMolDescriptors import CalcPBF, CalcRadiusOfGyration

import joblib
import pandas as pd
import numpy as np

import os

files_dir = "./mol_files_training/"
files_list_all = os.listdir(files_dir)
fiels_list = [file for file in files_list_all if file.endswith('.mol')]

#columns=["mol_weight", "mol_volume", "mol_polar_area", 
#                            "mol_surface_area", "logP", "num_rotatable_bonds", 
#                            "ring_count", "num_h_donors", 
#                            "num_h_acceptors",""]

def counter_electronegativity(mol):

    from rdkit import Chem
    from collections import Counter

    # 定义鲍林电负性表（只列出常见元素，可根据需要扩展）
    pauling_electronegativity = {
    'H': 2.20,  'He': None,  # 第一周期 
    'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None,  # 第二周期
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None,  # 第三周期
    'K': 0.82,  'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': None,  # 第四周期
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': None,  # 第五周期
    }

    # 获取所有原子的元素符号
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # 使用 Counter 统计每种元素的数量
    element_counts = Counter(atom_symbols)

    # 计算每种元素的电负性，并统计电负性大于 2.2 的元素数量
    high_electronegativity_count = 0
    for element, count in element_counts.items():
        electronegativity = pauling_electronegativity.get(element, None)
        
        if electronegativity is not None:
            # print(f"Element: {element}, Electronegativity: {electronegativity}, Count: {count}")
            
            # 统计电负性大于 2.55 的元素
            if electronegativity > 2.55:
                high_electronegativity_count += count

    # print(f"Total number of elements with electronegativity > 2.55: {high_electronegativity_count}")
    
    return high_electronegativity_count
    
    
def structurial_features(mol):
    
    # 获得分子中所有原子对的距离
    distance_matrix = AllChem.Get3DDistanceMatrix(mol)
    flattened_dist_matrix = distance_matrix.flatten() 
    
    diameter = np.max(distance_matrix)
   
    
    # 键长矩阵
    bond_lengths = [rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    bond_lengths_mean = np.mean(bond_lengths)
    
    # 键角矩阵
    bond_angles = [rdMolTransforms.GetAngleDeg(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetEndAtomIdx() + 1) for bond in mol.GetBonds() if bond.GetEndAtomIdx() < mol.GetNumAtoms()-1]
    bond_angles_mean = np.mean(bond_angles)
    
    # 二面角矩阵
    torsions = []
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        for bond2 in mol.GetBonds():
            if bond2.GetBeginAtomIdx() == atom2:
                atom3 = bond2.GetEndAtomIdx()
                for bond3 in mol.GetBonds():
                    if bond3.GetBeginAtomIdx() == atom3:
                        atom4 = bond3.GetEndAtomIdx()
                        torsion = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), atom1, atom2, atom3, atom4)
                        torsions.append(torsion)
    
    torsions_std = np.std(torsions)
    
    # 3D自相关系数
    autocorr = rdMolDescriptors.CalcAUTOCORR3D(mol)
    
    return diameter, bond_lengths_mean, bond_angles_mean, torsions_std
    
   

def generate_features_mol(mol):
    
    if mol is not None:
        
        df = pd.DataFrame()
        
        df.loc[0, 'Formula'] = rdMolDescriptors.CalcMolFormula(mol)
                            
        # 计算分子的分子量（分子质量）
        df.loc[0, 'mol_weight'] = Descriptors.MolWt(mol)
        
        # 计算分子的体积
        df.loc[0, 'mol_volume'] = AllChem.ComputeMolVolume(mol)
        
        # 计算分子的极性表面积
        df.loc[0, 'mol_polar_area'] = Descriptors.TPSA(mol)
        
        # 计算分子的平均表面积
        df.loc[0, 'mol_surface_area'] = Chem.rdMolDescriptors.CalcPBF(mol)
        
        # 计算分子的LogP值
        df.loc[0, 'logP'] = Descriptors.MolLogP(mol)
        
        # 计算分子中可旋转的键的数量
        df.loc[0, 'num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        
        # 计算分子中环的数量
        df.loc[0, 'ring_count'] = Descriptors.RingCount(mol)
        
        # 计算分子中氢键供体和受体的数量
        df.loc[0, 'num_h_donors'] = Descriptors.NumHDonors(mol)
        df.loc[0, 'num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
        
        # Calculate the Labute ASA (Accessible Surface Area)
        df.loc[0, 'surface_area'] = rdMolDescriptors.CalcLabuteASA(mol)
        
        # 计算每个原子的EState指数 (EState Indices)
        estate_indices = EState.EStateIndices(mol)
        
        df.loc[0, 'estate_indices_std'] = np.std(estate_indices)
        df.loc[0, 'estate_indices_sum'] = np.sum(estate_indices)
        df.loc[0, 'estate_indices_mean'] = np.mean(estate_indices)
        
        # 计算Balaban J描述符 (Balaban J Descriptor),描述符是衡量分子拓扑复杂性的一个重要工具
        df.loc[0, 'surface_area'] = GraphDescriptors.BalabanJ(mol)
        
        # 计算Bertz复杂性指数 (Bertz Complexity Index)
        df.loc[0, 'surface_area'] = Descriptors.BertzCT(mol)
        
        # 计算手性中心的数量 (Number of Chiral Centers)
        df.loc[0, 'chiral_centers'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # 计算回转半径 (Radius of Gyration), 分析分子中原子相对于质心（分子质量中心）的分布情况
        df.loc[0, 'radius_of_gyration'] = CalcRadiusOfGyration(mol)
        
        # 计算最佳平面拟合 (Plane of Best Fit, PBF) 反映分子最接近平面的程度。
        df.loc[0, 'pbf'] = CalcPBF(mol)   
        
        # 计算Chi0v和Chi1v描述符
        df.loc[0, 'chi0v'] = Chem.GraphDescriptors.Chi0v(mol)
        df.loc[0, 'chi1v'] = Chem.GraphDescriptors.Chi1v(mol)
        
        # 计算电负性大于C的原子的个数
        df.loc[0, 'counter_electronegativity'] = counter_electronegativity(mol)
                
        # structurial_features
        diameter, bond_lengths_mean, bond_angles_mean, torsions_std = structurial_features(mol)
        
        df.loc[0, 'diameter'] = diameter 
        df.loc[0, 'bond_lengths_mean'] = bond_lengths_mean
        df.loc[0, 'bond_angles_mean'] = bond_angles_mean
        df.loc[0, 'torsions_std'] = torsions_std 

    else:
        print(f"Error reading {mol}")

    return df

mol_list = []
smile_list = []
name_list = []
mol_error = []


mol_features = []

all_df = pd.DataFrame()
for index_i,i in enumerate(fiels_list):
    print(f"{index_i+1}/{len(fiels_list)}", i)
    mol = Chem.MolFromMolFile(files_dir + "/" + i)

    try:
        smilie_i = Chem.MolToSmiles(mol)
       
    except:
        print(i)
        mol_error.append(i)

    else:
        
        mol_list.append(i)
        
        smile_list.append(smilie_i)
        
        name_ = i.split(".")[0]
        name_list.append(int(name_))

        features = []

        df = generate_features_mol(mol)
        
        all_df = pd.concat([all_df,df],axis=0)
        
      
all_df.insert(0,"Smile", smile_list)
all_df.insert(0,"Filename", mol_list)
all_df.insert(0,"Name", name_list)

joblib.dump(mol_list,"mol_list.pkl")
joblib.dump(mol_features,"mol_features.pkl")


all_df.to_excel("features_training.xlsx")


