import itertools
import numpy as np
import pandas as pd
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class MoleculeQuantumStateGenerator():
    def __init__(
        self,
        heavy_atom_size=5,
        ncpus=4,
        sanitize_method="strict",
        stereo_chiral=True,
        atom_weights=None,
        atom_selection_bits=7,
    ):
        self.size = heavy_atom_size
        self.effective_numbers = list(range(heavy_atom_size))
        self.ncpus = ncpus

        # 預設支援九種重元素，順序固定以維持位元對應不變
        atom_sequence = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
        self.atom_type_to_idx = {atom: idx + 1 for idx, atom in enumerate(atom_sequence)}
        self.idx_to_atom_type = {idx + 1: atom for idx, atom in enumerate(atom_sequence)}

        default_weights = {
            "C": 25,
            "N": 25,
            "O": 25,
            "S": 5,
            "P": 5,
            "F": 5,
            "Cl": 5,
            "Br": 5,
            "I": 5,
        }
        self.atom_weights = atom_weights or default_weights
        # 僅保留支援的元素並依照 atom_sequence 的順序建立權重
        self.atom_weights = {
            atom: self.atom_weights.get(atom, default_weights[atom])
            for atom in atom_sequence
        }
        self.atom_selection_bits = atom_selection_bits

        self.bond_type_to_idx = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
        }
        self.idx_to_bond_type = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
        self.aromatic_bond_idx = self.bond_type_to_idx[Chem.rdchem.BondType.AROMATIC]
        # 量測位元長度可依需求調整
        self.qubits_per_type_atom = atom_selection_bits
        self.qubits_per_type_bond = int(
            np.ceil(np.log2(len(self.bond_type_to_idx) + 1))
        )
        self.n_qubits = int(
            self.size * self.qubits_per_type_atom
            + self.size * (self.size - 1) / 2 * self.qubits_per_type_bond
        )
        self.sanitize_method = sanitize_method
        self.atom_valence_dict = {
            "C": 4,
            "N": 3,
            "O": 2,
            "S": 6,
            "P": 5,
            "F": 1,
            "Cl": 1,
            "Br": 1,
            "I": 1,
        }
        self.stereo_chiral = stereo_chiral

        # 將權重轉換成累積分佈，供後續位元值對應元素
        self._build_atom_weight_bins()

    def _build_atom_weight_bins(self):
        self.total_atom_weight = sum(self.atom_weights.values())
        cumulative = 0
        self.atom_weight_bins = []
        self.atom_weight_encoding = {}
        for atom, weight in self.atom_weights.items():
            start = cumulative
            cumulative += weight
            self.atom_weight_bins.append((cumulative, atom))
            self.atom_weight_encoding[atom] = (start, cumulative)

    def _select_atom_from_value(self, value):
        if self.total_atom_weight == 0:
            return None
        mod_value = value % self.total_atom_weight
        # 將位元轉換的整數映射回累積權重區間，取得原子符號
        for upper, atom in self.atom_weight_bins:
            if mod_value < upper:
                return atom
        return None

    def _encode_atom_to_value(self, atom):
        if atom not in self.atom_weight_encoding:
            return 0
        start, _ = self.atom_weight_encoding[atom]
        return start

    def decimal_to_binary(self, x, padding_length=None):
        """
        Parameters:
        x (int): The decimal value.

        Returns:
        str: A binary bit string.
        """
        if padding_length is None:
            padding_length = self.qubits_per_type_atom
        # 以固定長度補零，確保位元排列一致
        bit = "0" * (padding_length - 1) + bin(x)[2:]
        return bit[-padding_length:]

    def SmilesToConnectivity(self, smiles):
        """
        Generate a molecular graph from a SMILES string.

        Parameters:
        smiles (str): The SMILES string representing the molecule.

        Returns:
        tuple: A tuple containing the node vector (np.ndarray) and the adjacency matrix (np.ndarray).
        """
        node_vector = np.zeros(self.size)
        adjacency_matrix = np.zeros((self.size, self.size))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return node_vector, adjacency_matrix
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_type = atom.GetSymbol()
            node_vector[idx] = self.atom_type_to_idx[atom_type]
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if bond.GetIsAromatic():
                bond_type_idx = self.aromatic_bond_idx
            else:
                bond_type_idx = self.bond_type_to_idx.get(bond.GetBondType(), 0)
            adjacency_matrix[i][j] = bond_type_idx
            adjacency_matrix[j][i] = bond_type_idx
        return node_vector, adjacency_matrix

    def _rank_list(self, lst):
        sorted_list = sorted(enumerate(lst), key=lambda x: x[1])
        rank = [0] * len(lst)
        for i, (original_index, _) in enumerate(sorted_list):
            rank[original_index] = i + 1
        return rank

    def _can_sort_with_even_swaps(self, list1, list2):
        def count_inversions(lst):
            inversions = 0
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if lst[i] > lst[j]:
                        inversions += 1
            return inversions
        inversions_list1 = count_inversions(list1)
        inversions_list2 = count_inversions(list2)
        return (inversions_list1 - inversions_list2) % 2 == 0

    def _set_chiral_atom(self, mol):
        """ Based on the atom-mapping and CIP information to determine the R/S chirality. """
        for atom in mol.GetAtoms():
            if atom.GetPropsAsDict(True, False).get("_ChiralityPossible", 0):
                atom_map_list = [int(neighbor.GetProp('molAtomMapNumber')) for neighbor in atom.GetNeighbors()]
                CIP_list = [int(neighbor.GetProp('_CIPRank')) for neighbor in atom.GetNeighbors()]
                chiral_tag = self._can_sort_with_even_swaps(self._rank_list(atom_map_list), self._rank_list(CIP_list))
                if chiral_tag:
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                else:
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)

    def _determine_bond_stereo(self, bond):
        begin_atom = bond.GetBeginAtom()
        begin_atom_map_number = int(begin_atom.GetProp('molAtomMapNumber'))
        end_atom = bond.GetEndAtom()
        end_atom_map_number = int(end_atom.GetProp('molAtomMapNumber'))
        begin_atom_neighbor_map = [int(a.GetProp('molAtomMapNumber')) for a in begin_atom.GetNeighbors()]
        begin_atom_neighbor_map.remove(end_atom_map_number)
        end_atom_neighbor_map = [int(a.GetProp('molAtomMapNumber')) for a in end_atom.GetNeighbors()]
        end_atom_neighbor_map.remove(begin_atom_map_number)
        if (len(begin_atom_neighbor_map) == 1) and (len(end_atom_neighbor_map) == 1):
            if abs(begin_atom_neighbor_map[0] - begin_atom_map_number) == abs(end_atom_neighbor_map[0] - end_atom_map_number):
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            else:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
        else:
            begin_CIP_list = [
                int(neighbor.GetProp('_CIPRank'))
                for neighbor in begin_atom.GetNeighbors()
                if int(neighbor.GetProp('molAtomMapNumber')) != end_atom_map_number
            ]
            end_CIP_list = [
                int(neighbor.GetProp('_CIPRank'))
                for neighbor in end_atom.GetNeighbors()
                if int(neighbor.GetProp('molAtomMapNumber')) != begin_atom_map_number
            ]
            begin_even = self._can_sort_with_even_swaps(
                self._rank_list(begin_atom_neighbor_map),
                self._rank_list(begin_CIP_list),
            )
            end_even = self._can_sort_with_even_swaps(
                self._rank_list(end_atom_neighbor_map),
                self._rank_list(end_CIP_list),
            )
            if begin_even == end_even:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            else:
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
        return

    def _set_stereo_bond(self, mol):
        Chem.FindPotentialStereoBonds(mol,cleanIt=True)
        for bond in mol.GetBonds():
            if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOANY:
                self._determine_bond_stereo(bond)
        return

    def ConnectivityToSmiles(self, node_vector, adjacency_matrix):
        """
        Generate a SMILES string from the molecular graph.

        Returns:
        str: The SMILES string representing the molecule.
        """
        mol = Chem.RWMol()
        mapping_num_2_molIdx = {}
        # 為每個非零原子建立 RDKit 原子並保留原始索引
        for i, atom_type_idx in enumerate(node_vector):
            if atom_type_idx == 0:
                continue
            a = Chem.Atom(self.idx_to_atom_type[atom_type_idx])
            a.SetAtomMapNum(i+1)
            molIdx = mol.AddAtom(a)
            mapping_num_2_molIdx.update({i: molIdx})
        # add bonds between adjacent atoms, only traverse half the matrix
        uses_aromatic = False
        for ix, row in enumerate(adjacency_matrix):
            for iy_, bond_type_idx in enumerate(row[ix+1:]):
                iy = ix + iy_ + 1
                if bond_type_idx == 0:
                    continue
                else:
                    bond_type = self.idx_to_bond_type.get(bond_type_idx)
                    if bond_type is None:
                        continue
                    aromatic_edge = bond_type == Chem.rdchem.BondType.AROMATIC
                    if aromatic_edge:
                        uses_aromatic = True
                    try:
                        mol.AddBond(mapping_num_2_molIdx[ix], mapping_num_2_molIdx[iy], bond_type)
                    except:
                        return None
                    if aromatic_edge:
                        bond = mol.GetBondBetweenAtoms(mapping_num_2_molIdx[ix], mapping_num_2_molIdx[iy])
                        if bond is None:
                            return None
                        bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
                        bond.SetIsAromatic(True)
                        begin_atom = mol.GetAtomWithIdx(mapping_num_2_molIdx[ix])
                        end_atom = mol.GetAtomWithIdx(mapping_num_2_molIdx[iy])
                        begin_atom.SetIsAromatic(True)
                        end_atom.SetIsAromatic(True)
        mol = mol.GetMol()
        if self.sanitize_method == "strict":
            try:
                if uses_aromatic:
                    sanitize_ops = (
                        Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    )
                    Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
                    Chem.SetAromaticity(mol)
                else:
                    Chem.SanitizeMol(mol)
            except:
                return None
        elif self.sanitize_method == "soft":
            try:
                if uses_aromatic:
                    sanitize_ops = (
                        Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    )
                    Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
                    Chem.SetAromaticity(mol)
                else:
                    Chem.SanitizeMol(mol)
            except:
                try:
                    for atom in mol.GetAtoms():
                        bond_count = int(sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]))
                        if atom.GetSymbol() == "O" and bond_count >= 3:
                            atom.SetFormalCharge(bond_count - 2)
                        elif atom.GetSymbol() == "N" and bond_count >= 4:
                            atom.SetFormalCharge(bond_count - 3)
                        elif atom.GetSymbol() == "c" and bond_count >= 5:
                            atom.SetFormalCharge(bond_count - 4)
                    if uses_aromatic:
                        sanitize_ops = (
                            Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                        )
                        Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
                        Chem.SetAromaticity(mol)
                    else:
                        Chem.SanitizeMol(mol)
                except:
                    return None

        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
        if self.stereo_chiral:
            self._set_chiral_atom(mol)
            self._set_stereo_bond(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)

    def ConnectivityToQuantumState(self, node_vector, adjacency_matrix):
        """
        Generate the quantum state (bit vector) based on the molecular connectivity.
        The preceding bits represent the atom type, and the subsequent bits represent the connectivity.

        Returns:
        np.ndarray: computational quantum state.
        """
        quantum_state = ""
        for atom_idx in node_vector:
            if int(atom_idx) == 0:
                value = 0
            else:
                atom_symbol = self.idx_to_atom_type.get(int(atom_idx))
                value = self._encode_atom_to_value(atom_symbol)
            quantum_state += self.decimal_to_binary(
                int(value), padding_length=self.qubits_per_type_atom
            )
        # 原子片段之後依序放入鍵的型別位元
        for ix, row in enumerate(adjacency_matrix):
            for bond_type_idx in row[ix+1:]:
                quantum_state += self.decimal_to_binary(int(bond_type_idx), padding_length=self.qubits_per_type_bond)
        return quantum_state

    def QuantumStateToConnectivity(self, quantum_state):
        # 依照原本配置拆解原子位元數，再取出鍵位元區塊
        node_bits = self.size * self.qubits_per_type_atom
        node_state = quantum_state[:node_bits]
        bond_state = quantum_state[node_bits:]
        node_vector = np.zeros(self.size)
        adjacency_matrix = np.zeros((self.size, self.size))
        for i in range(self.size):
            start = i * self.qubits_per_type_atom
            atom_bits = node_state[start : start + self.qubits_per_type_atom]
            atom_value = int(atom_bits, 2)
            atom_symbol = self._select_atom_from_value(atom_value)
            if atom_symbol:
                node_vector[i] = self.atom_type_to_idx[atom_symbol]
        bond_pairs = [
            (i, j) for i in range(self.size - 1) for j in range(i + 1, self.size)
        ]
        for idx, (row, column) in enumerate(bond_pairs):
            start = idx * self.qubits_per_type_bond
            bond_bits = bond_state[start : start + self.qubits_per_type_bond]
            bond_type_idx = int(bond_bits, 2)
            if bond_type_idx not in self.idx_to_bond_type:
                bond_type_idx = 0
            adjacency_matrix[row][column] = bond_type_idx
            adjacency_matrix[column][row] = bond_type_idx
        return node_vector, adjacency_matrix

    def QuantumStateToSmiles(self, quantum_state):
        return self.ConnectivityToSmiles(*self.QuantumStateToConnectivity(quantum_state))

    def QuantumStateToStateVector(self, quantum_state):
        stat_vector = np.zeros(2**self.n_qubits)
        decimal = int(quantum_state, 2)
        stat_vector[-1-decimal] = 1
        return stat_vector

    def QuantumStateToDecimal(self, quantum_state):
        decimal = int(quantum_state, 2)
        return decimal

    def post_process_quantum_state(self, result_state: str, reverse=True):
        """
        Reverse the qiskit outcome state and change the order to meet the definition of node vector and adjacency matrix.

        :param result_state: computational state derived from qiskit measurement outcomes
        :return: str of post-processed quantum state
        """
        expected_length = int(self.n_qubits)
        assert len(result_state) == expected_length
        if reverse:
            result_state = result_state[::-1]
        quantum_state = ""
        idx = 0
        for _ in range(self.size):
            quantum_state += result_state[idx : idx + self.qubits_per_type_atom]
            idx += self.qubits_per_type_atom
        num_bonds = int(self.size * (self.size - 1) / 2)
        # 逐段擷取鍵的位元資訊，重建原始排列
        for _ in range(num_bonds):
            quantum_state += result_state[idx : idx + self.qubits_per_type_bond]
            idx += self.qubits_per_type_bond
        return quantum_state

    def generate_permutations(self, k):
        """
        Generate all possible permutations of k elements from the given list of elements.

        :param k: Number of elements to choose for permutations
        :return: List of permutations
        """
        return list(itertools.permutations(self.effective_numbers, k))

    def enumerate_all_quantum_states(self, smiles):
        """
        Generate all possible quantum states representing the given molecule SMILES.

        :return: List of quantum states (str)
        """
        node_vector, adjacency_matrix = self.SmilesToConnectivity(smiles)
        all_permutation_index = self.generate_permutations(np.count_nonzero(node_vector))
        args = [(self, node_vector, adjacency_matrix, new_index) for new_index in all_permutation_index]
        with multiprocessing.Pool(processes=self.ncpus) as pool:
            all_quantum_states = pool.starmap(subfunction_generate_state, args)

        return list(set(all_quantum_states))

    def permutate_connectivity(self, node_vector, adjacency_matrix, new_index):
        mapping_dict = {old: new for old, new in enumerate(new_index)}
        new_node_vector = np.zeros(self.size)
        new_adjacency_matrix = np.zeros((self.size, self.size))
        # atom
        for old, new in mapping_dict.items():
            new_node_vector[new] = node_vector[old]
        # bond
        for ix, row in enumerate(adjacency_matrix):
            for iy_, bond_type_idx in enumerate(row[ix+1:]):
                if not bond_type_idx:
                    continue
                iy = ix + iy_ + 1
                ix_new = mapping_dict[ix]
                iy_new = mapping_dict[iy]
                new_adjacency_matrix[ix_new][iy_new] = bond_type_idx
                new_adjacency_matrix[iy_new][ix_new] = bond_type_idx
        return new_node_vector, new_adjacency_matrix

    def generate_valid_mask(self, data: pd.DataFrame):
        """
        :return: binary valid quantum states mask (np.ndarray)
        """
        valid_state_vector_mask = np.zeros(2**self.n_qubits)
        for decimal_index in set(data["decimal_index"]):
            # 將資料集中出現的態標記為有效
            valid_state_vector_mask[int(decimal_index)] = 1
        return valid_state_vector_mask

def draw_top_molecules(smiles_dict, top_n=10, mols_per_row=5, mol_size=(300, 300)):
    """
    Draw the top N molecules with highest frequency from SMILES dictionary
    
    Parameters:
    - smiles_dict: Dictionary containing SMILES strings and their frequencies
    - top_n: Number of molecules to display, default is 10
    - mols_per_row: Number of molecules per row, default is 5
    - mol_size: Size of each molecule image, default is (300, 300)
    """
    
    # 移除無法產生 SMILES 的條目，避免後續繪圖失敗
    filtered_dict = {k: v for k, v in smiles_dict.items() if k is not None}
    sorted_smiles = sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    molecules = []
    legends = []
    
    for smiles, count in sorted_smiles:
        mol = Chem.MolFromSmiles(smiles)
        

        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        
        molecules.append(mol)
        legends.append(f"{smiles}\nCount: {count}")

    drawer_opts = rdMolDraw2D.MolDrawOptions()
    drawer_opts.addStereoAnnotation = True 
    drawer_opts.addAtomIndices = False     
    drawer_opts.bondLineWidth = 2           
    drawer_opts.multipleBondOffset = 0.15  
    
    img = Draw.MolsToGridImage(
        molecules, 
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        legends=legends,
        useSVG=True,
        drawOptions=drawer_opts,
    )
    display(img)
