import numpy as np
import math
import random
from itertools import permutations
import torch
from .utils import *

# For reproducibility only
def set_seed(
    RANDOM_SEED: int = 42
):
    '''
    Set the random seed for reproducibility.
    
    Args:
        RANDOM_SEED (int): The random seed to set.
    '''
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def split_trDR_teDR_Px(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    rotation_bank: int = 2,
    color_bank: int = 3,
    DA_dataset_scaling: float = 1.5,
    DA_epoch_locker_num: int = 10,
    DA_random_locker: bool = False,
    DA_max_dist: int = 2,
    DA_continual_divergence: bool = False,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A-A-B-B-B-C-C-C-A-A-D-D
        Testing: E (unseen)
        with distribution difference in P(x)
        for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        rotation_bank (int): The number of rotation patterns. 1 as no rotation.
        color_bank (int): The number of color patterns. 1 as no color.
        [DA_parameters]: Details in Description.
        verbose (bool): Whether to print the distribution information.

    Description:
        A pattern bank will be created based on #rotation and #color.
        Each pattern is considered as a type of P(x) distribution (A/B/C ...).
        Each subset taking a certain pattern for all its datapoints.
        
        DA_dataset_scaling (float): The scaling factor for the training dataset.
            (size = original size * dataset_scaling)
        DA_epoch_locker_num (int): The number of dataset changing stages during overall training.
            E.g., 2 means the dataset changes once (two stages), and an indicator float=0.5 will be labeled
            to the subdataset to let known when to change.
            Accordingly, 3 gives float=0.33,0.67.
        DA_random_locker (bool): Whether clients are randomly assigned to epoch lockers.
            When true, the locker float is randomly generated.
            E.g., when DA_epoch_locker_num=3 with DA_random_locker=True, the locker floats could be 0.20,0.80.
        DA_max_dist (int): The maximum dist types during overall training.
            E.g., when DA_epoch_locker_num=5 with DA_max_dist=3,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. 3=(A,B,C)
            drifting as [A]-[B]-[C]-[C]-[D] is INVALID. 4=(A,B,C,D)
        DA_continual_divergence (bool): Whether the distribution drifts continually.
            E.g. when True,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. (continual)
            drifting as [A]-[B]-[B]-[C]-[A] is INVALID. (back to dist A)
        
    Warning:
        EXTENSION: YES. Datapoints replicated overall. (by dataset_scaling)
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

        VERBOSE is always recommended for sanity check.

        Set DA_dataset_scaling pairing your DA_epoch_locker_num
            when DA_dataset_scaling==DA_epoch_locker_num, #datapoints same as original.

        #TODO to support manual distribution assignment.

    Returns:
        A list of dictionaries contains:
            train, (bool) whether it is training set
            dataset,
            client number,
            epoch locker indicator, (float) to tell the changing time
            epoch locker order: (int) to tell the order of current subset
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 0 < rotation_bank, "The number of rotation patterns must be greater than 0."
    assert 0 < color_bank, "The number of color patterns must be greater than 0."
    assert DA_dataset_scaling >= 1, "Invalid downscaling."
    assert DA_epoch_locker_num > 0, "The number of epoch lockers must be greater than 0."
    assert 1 <= DA_max_dist, "Distribution assignment out of range."
    DA_max_dist = min(DA_max_dist, rotation_bank * color_bank)

    # generate pattern bank
    angles = [i * 360 / rotation_bank for i in range(rotation_bank)] if rotation_bank > 1 else [0.0]

    if color_bank == 1:
        colors = ['gray']
    elif color_bank == 2:
        colors = ['red', 'blue']
    elif color_bank == 3:
        colors = ['red', 'blue', 'green']
    else:
        raise ValueError("The number of color patterns must be 1, 2, or 3.")

    pattern_bank = {i + 1: [angle, color] for i, (angle, color)
                    in enumerate([(angle, color) for angle in angles for color in colors])}
    
    print("Pattern bank:\n", '\n'.join(f"{key}: {value}" for key, value in pattern_bank.items())) if verbose else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []
    client_Count = 0

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None
        # training dataset scaling
        original_train_feature = client_data_train['features']
        original_train_label = client_data_train['labels']
        indices = torch.randint(0, original_train_label.shape[0],
                                (int(original_train_label.shape[0] * (DA_dataset_scaling - 1)),))
        sampled_data = original_train_feature[indices]
        sampled_label = original_train_label[indices]

        cur_train_feature = torch.cat((original_train_feature, sampled_data), dim=0)
        cur_train_label = torch.cat((original_train_label, sampled_label), dim=0)
        permuted_indices = torch.randperm(cur_train_label.shape[0])
        cur_train_feature = cur_train_feature[permuted_indices]
        cur_train_label = cur_train_label[permuted_indices]

        cur_test_feature = client_data_test['features']
        cur_test_label = client_data_test['labels']

        # generate drifting
        dist_bank = list(range(1, rotation_bank * color_bank + 1))
        test_dist = np.random.choice(dist_bank)
        dist_bank.remove(test_dist)
        train_dist = generate_DA_dist(dist_bank,
                                      DA_epoch_locker_num,DA_max_dist,DA_continual_divergence)
        
        lockers = sorted(torch.rand(DA_epoch_locker_num - 1).tolist() + [0.0]) if DA_random_locker \
                else torch.linspace(0, 1, steps=DA_epoch_locker_num + 1)[:-1].tolist()

        print("Train distribution: ", train_dist,
              "\nTest distribution: ", test_dist,
              "\nEpoch lockers: ", lockers,
              "\n") if verbose else None

        # training subsets
        feature_chunks = torch.chunk(cur_train_feature, DA_epoch_locker_num, dim=0)
        label_chunks = torch.chunk(cur_train_label, DA_epoch_locker_num, dim=0)

        # Loop through feature chunks and label chunks
        for i, (feature_chunk, label_chunk) in enumerate(zip(feature_chunks, label_chunks)):
            # Get angle and color from the pattern bank based on the train_dist
            angle, color = pattern_bank[train_dist[i]]

            # Apply rotation and color transformations
            feature_chunk = rotate_dataset(feature_chunk, [float(angle)] * feature_chunk.shape[0])
            feature_chunk = color_dataset(feature_chunk, [color] * feature_chunk.shape[0])

            # Append the cumulative data to rearranged_data
            rearranged_data.append({
                'train': True,
                'features': feature_chunk.detach().cpu().numpy(),
                'labels': label_chunk.detach().cpu().numpy(),
                'client_number': client_Count,
                'epoch_locker_indicator': lockers[i],
                'epoch_locker_order': i,
                'cluster': train_dist[i]
            })

        # testing set
        angle, color = pattern_bank[test_dist]
        cur_test_feature = rotate_dataset(cur_test_feature, [float(angle)] * cur_test_feature.shape[0])
        cur_test_feature = color_dataset(cur_test_feature, [color] * cur_test_feature.shape[0])

        rearranged_data.append({
            'train': False,
            'features': cur_test_feature.detach().cpu().numpy(),
            'labels': cur_test_label.detach().cpu().numpy(),
            'client_number': client_Count,
            'epoch_locker_indicator': -1.0,
            'epoch_locker_order': -1,
            'cluster': test_dist
        })

        client_Count += 1

    return rearranged_data

def split_trDR_teDR_Py(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    py_bank: int = 10,
    classes_per_set: int = 2,
    DA_dataset_scaling: float = 1.5,
    DA_epoch_locker_num: int = 10,
    DA_random_locker: bool = False,
    DA_max_dist: int = 2,
    DA_continual_divergence: bool = False,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A-A-B-B-B-C-C-C-A-A-D-D
        Testing: E (unseen)
        with distribution difference in P(y)
        for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        py_bank (int): The number of diff py dists.
        classes_per_set (int): The number of classes per subdataset.
        [DA_parameters]: Details in Description.
        verbose (bool): Whether to print the distribution information.

    Description:
        One subset has classes_per_set classes as a type of P(y) distribution (A/B/C ...).
        
        DA_dataset_scaling (float): The scaling factor for the training dataset.
            (size = original size * dataset_scaling)
        DA_epoch_locker_num (int): The number of dataset changing stages during overall training.
            E.g., 2 means the dataset changes once (two stages), and an indicator float=0.5 will be labeled
            to the subdataset to let known when to change.
            Accordingly, 3 gives float=0.33,0.67.
        DA_random_locker (bool): Whether clients are randomly assigned to epoch lockers.
            When true, the locker float is randomly generated.
            E.g., when DA_epoch_locker_num=3 with DA_random_locker=True, the locker floats could be 0.20,0.80.
        DA_max_dist (int): The maximum dist types during overall training.
            E.g., when DA_epoch_locker_num=5 with DA_max_dist=3,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. 3=(A,B,C)
            drifting as [A]-[B]-[C]-[C]-[D] is INVALID. 4=(A,B,C,D)
        DA_continual_divergence (bool): Whether the distribution drifts continually.
            E.g. when True,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. (continual)
            drifting as [A]-[B]-[B]-[C]-[A] is INVALID. (back to dist A)
        
    Warning:
        EXTENSION: YES. Datapoints replicated overall. (by dataset_scaling)
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

        VERBOSE is always recommended for sanity check.

        #TODO to support manual distribution assignment.

    Returns:
        A list of dictionaries contains:
            train, (bool) whether it is training set
            dataset,
            client number,
            epoch locker indicator, (float) to tell the growth time
            epoch locker order: (int) to tell the order of current subset
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert DA_dataset_scaling >= 1, "Invalid downscaling."
    assert DA_epoch_locker_num > 0, "The number of epoch lockers must be greater than 0."
    assert 1 <= DA_max_dist, "Distribution assignment out of range."
    DA_max_dist = min(DA_max_dist, py_bank)
    assert torch.unique(train_labels).size(0) == torch.unique(test_labels).size(0), "Original Dataset Fault."
    label_num = torch.unique(train_labels).size(0)
    assert 1 <= classes_per_set <= label_num, "Invalid number of classes per set."

    # generate py bank
    py_class_bank = {i: sorted(np.random.choice(label_num, classes_per_set, replace=False).tolist())
                   for i in range(1, py_bank + 1)}

    print("Py bank:\n", '\n'.join(f"{key}: {value}" for key, value in py_class_bank.items())) if verbose else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []
    client_Count = 0

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None
        # training dataset scaling
        # not supporting scaling for now

        cur_train_feature = client_data_train['features']
        cur_train_label = client_data_train['labels']
        cur_test_feature = client_data_test['features']
        cur_test_label = client_data_test['labels']

        # generate drifting
        dist_bank = list(range(1, py_bank + 1))
        test_dist = np.random.choice(dist_bank)
        dist_bank.remove(test_dist)
        train_dist = generate_DA_dist(dist_bank,
                                      DA_epoch_locker_num,DA_max_dist,DA_continual_divergence)
        
        lockers = sorted(torch.rand(DA_epoch_locker_num - 1).tolist() + [0.0]) if DA_random_locker \
                else torch.linspace(0, 1, steps=DA_epoch_locker_num + 1)[:-1].tolist()

        print("Train distribution: ", train_dist,
              "\nTest distribution: ", test_dist,
              "\nEpoch lockers: ", lockers,
              "\n") if verbose else None

        # training subsets
        for i, dist in enumerate(train_dist):
            locker_classes_tensor = torch.tensor(py_class_bank[dist])
            mask = torch.isin(cur_train_label, locker_classes_tensor)

            # Filter the data
            filtered_train_feature = cur_train_feature[mask]
            filtered_train_label = cur_train_label[mask]

            # Append the cumulative data to rearranged_data
            rearranged_data.append({
                'train': True,
                'features': filtered_train_feature.detach().cpu().numpy(),
                'labels': filtered_train_label.detach().cpu().numpy(),
                'client_number': client_Count,
                'epoch_locker_indicator': lockers[i],
                'epoch_locker_order': i,
                'cluster': train_dist[i]
            })

        # testing set
        locker_classes_tensor = torch.tensor(py_class_bank[test_dist])
        mask = torch.isin(cur_test_label, locker_classes_tensor)

        filtered_test_feature = cur_test_feature[mask]
        filtered_test_label = cur_test_label[mask]

        rearranged_data.append({
            'train': False,
            'features': filtered_test_feature.detach().cpu().numpy(),
            'labels': filtered_test_label.detach().cpu().numpy(),
            'client_number': client_Count,
            'epoch_locker_indicator': -1.0,
            'epoch_locker_order': -1,
            'cluster': test_dist
        })

        client_Count += 1

    return rearranged_data

def split_trDR_teDR_Py_x(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    mixing_num: int = 3,
    DA_dataset_scaling: float = 1.5,
    DA_epoch_locker_num: int = 10,
    DA_random_locker: bool = False,
    DA_max_dist: int = 2,
    DA_continual_divergence: bool = False,
    verbose: bool = True
) -> list:
    """
    Split the dataset into distributions as:
        Training: A-A-B-B-B-C-C-C-A-A-D-D
        Testing: E (unseen)
        with distribution difference in P(y|x)
        for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        mixing_num (int): The number of mixed labels.
        [DA_parameters]: Details in Description.
        verbose (bool): Whether to print the distribution information.

    Description:
        #mixing_num classes will be randomly selected for label-swapping.
        One permutation of classes as a type of P(x|y) distribution (A/B/C ...).
        E.g. mixing_num=3, classes [0,5,7] are selected. The swap_bank will be:
        [0,5,7],[0,7,5],[5,0,7],[5,7,0],[7,0,5],[7,5,0]
        
        DA_dataset_scaling (float): The scaling factor for the training dataset.
            (size = original size * dataset_scaling)
        DA_epoch_locker_num (int): The number of dataset growth stages during overall training.
            E.g., 2 means the dataset grows once (two stages), and an indicator float=0.5 will be labeled
            to the subdataset to let known when to grow.
            Accordingly, 3 gives float=0.33,0.67.
        DA_random_locker (bool): Whether clients are randomly assigned to epoch lockers.
            When true, the locker float is randomly generated.
            E.g., when DA_epoch_locker_num=3 with DA_random_locker=True, the locker floats could be 0.20,0.80.
        DA_max_dist (int): The maximum dist types during overall training.
            E.g., when DA_epoch_locker_num=5 with DA_max_dist=3,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. 3=(A,B,C)
            drifting as [A]-[B]-[C]-[C]-[D] is INVALID. 4=(A,B,C,D)
        DA_continual_divergence (bool): Whether the distribution drifts continually.
            E.g. when True,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. (continual)
            drifting as [A]-[B]-[B]-[C]-[A] is INVALID. (back to dist A)
        
    Warning:
        EXTENSION: YES. Datapoints replicated overall. (by dataset_scaling)
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

        VERBOSE is always recommended for sanity check.

        Set DA_dataset_scaling pairing your DA_epoch_locker_num
            when DA_dataset_scaling==DA_epoch_locker_num, #datapoints same as original.

        #TODO to support manual classes assignment.

    Returns:
        A list of dictionaries contains:
            train, (bool) whether it is training set
            dataset,
            client number,
            epoch locker indicator, (float) to tell the growth time
            epoch locker order: (int) to tell the order of current subset
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert DA_dataset_scaling >= 1, "Invalid downscaling."
    assert DA_epoch_locker_num > 0, "The number of epoch lockers must be greater than 0."
    assert 1 <= DA_max_dist, "Distribution assignment out of range."
    DA_max_dist = min(DA_max_dist, math.factorial(mixing_num))
    assert torch.unique(train_labels).size(0) == torch.unique(test_labels).size(0), "Original Dataset Fault."
    label_num = torch.unique(train_labels).size(0)
    assert 1 <= mixing_num <= label_num, "Mixing class number out of range."

    # generate swapping bank
    class_list = sorted(np.random.choice(label_num, mixing_num, replace=False).tolist())
    all_permutations = list(permutations(class_list))

    swapping_bank = {i+1: {class_list[j]: perm[j] for j in range(mixing_num)}
                     for i, perm in enumerate(all_permutations)}

    print("Swapping bank:\n", '\n'.join(f"{key}: {value}" for key, value in swapping_bank.items())) if verbose else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []
    client_Count = 0

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None
        # training dataset scaling
        original_train_feature = client_data_train['features']
        original_train_label = client_data_train['labels']
        indices = torch.randint(0, original_train_label.shape[0],
                                (int(original_train_label.shape[0] * (DA_dataset_scaling - 1)),))
        sampled_data = original_train_feature[indices]
        sampled_label = original_train_label[indices]

        cur_train_feature = torch.cat((original_train_feature, sampled_data), dim=0)
        cur_train_label = torch.cat((original_train_label, sampled_label), dim=0)
        permuted_indices = torch.randperm(cur_train_label.shape[0])
        cur_train_feature = cur_train_feature[permuted_indices]
        cur_train_label = cur_train_label[permuted_indices]

        cur_test_feature = client_data_test['features']
        cur_test_label = client_data_test['labels']

        # generate drifting
        dist_bank = list(range(1, math.factorial(mixing_num) + 1))
        test_dist = np.random.choice(dist_bank)
        dist_bank.remove(test_dist)
        train_dist = generate_DA_dist(dist_bank,
                                      DA_epoch_locker_num,DA_max_dist,DA_continual_divergence)
        
        lockers = sorted(torch.rand(DA_epoch_locker_num - 1).tolist() + [0.0]) if DA_random_locker \
                else torch.linspace(0, 1, steps=DA_epoch_locker_num + 1)[:-1].tolist()

        print("Train distribution: ", train_dist,
              "\nTest distribution: ", test_dist,
              "\nEpoch lockers: ", lockers,
              "\n") if verbose else None

        # training subsets
        feature_chunks = torch.chunk(cur_train_feature, DA_epoch_locker_num, dim=0)
        label_chunks = torch.chunk(cur_train_label, DA_epoch_locker_num, dim=0)

        # Loop through feature chunks and label chunks
        for i, (feature_chunk, label_chunk) in enumerate(zip(feature_chunks, label_chunks)):
            # Get remapping
            label_remapping = swapping_bank[train_dist[i]]

            # Label swapping
            remapped_label_chunk = torch.clone(label_chunk)
            for original_label, new_label in label_remapping.items():
                remapped_label_chunk[label_chunk == original_label] = new_label

            # Append the cumulative data to rearranged_data
            rearranged_data.append({
                'train': True,
                'features': feature_chunk.detach().cpu().numpy(),
                'labels': remapped_label_chunk.detach().cpu().numpy(),
                'client_number': client_Count,
                'epoch_locker_indicator': lockers[i],
                'epoch_locker_order': i,
                'cluster': train_dist[i]
            })

        # Testing set
        label_remapping = swapping_bank[test_dist]
        remapped_label = torch.clone(cur_test_label)
        for original_label, new_label in label_remapping.items():
            remapped_label[cur_test_label == original_label] = new_label

        rearranged_data.append({
            'train': False,
            'features': cur_test_feature.detach().cpu().numpy(),
            'labels': remapped_label.detach().cpu().numpy(),
            'client_number': client_Count,
            'epoch_locker_indicator': -1.0,
            'epoch_locker_order': -1,
            'cluster': test_dist
        })

        client_Count += 1

    return rearranged_data

def split_trDR_teDR_Px_y(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    rotation_bank: int = 2,
    color_bank: int = 2,
    pyx_pattern_bank_num: int = 10,
    targeted_class_number: int = 2,
    DA_dataset_scaling: float = 1.5,
    DA_epoch_locker_num: int = 10,
    DA_random_locker: bool = False,
    DA_max_dist: int = 2,
    DA_continual_divergence: bool = False,
    verbose: bool = True
) -> list:
    """
    Split the dataset into distributions as:
        Training: A-A-B-B-B-C-C-C-A-A-D-D
        Testing: E (unseen)
        with distribution difference in P(x|y)
        for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        rotation_bank (int): The number of rotation patterns. 1 as no rotation.
        color_bank (int): The number of color patterns. 1 as no color.
        pyx_pattern_bank_num (int): The number of diff pattern dists.
        targeted_class_number (int): The number of classes applied augmentation per set.
        [DA_parameters]: Details in Description.
        verbose (bool): Whether to print the distribution information.

    Description:
        A Px pattern bank will be created based on #rotation and #color.
        A Pyx pattern bank (len=#pyx_pattern_bank_num) will be created as a type of P(y|x) distribution (A/B/C ...).
        Each pattern chooses a Px pattern and applies to #targeted_class_number classes (randomly selected).
        
        DA_dataset_scaling (float): The scaling factor for the training dataset.
            (size = original size * dataset_scaling)
        DA_epoch_locker_num (int): The number of dataset growth stages during overall training.
            E.g., 2 means the dataset grows once (two stages), and an indicator float=0.5 will be labeled
            to the subdataset to let known when to grow.
            Accordingly, 3 gives float=0.33,0.67.
        DA_random_locker (bool): Whether clients are randomly assigned to epoch lockers.
            When true, the locker float is randomly generated.
            E.g., when DA_epoch_locker_num=3 with DA_random_locker=True, the locker floats could be 0.20,0.80.
        DA_max_dist (int): The maximum dist types during overall training.
            E.g., when DA_epoch_locker_num=5 with DA_max_dist=3,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. 3=(A,B,C)
            drifting as [A]-[B]-[C]-[C]-[D] is INVALID. 4=(A,B,C,D)
        DA_continual_divergence (bool): Whether the distribution drifts continually.
            E.g. when True,
            drifting as [A]-[B]-[B]-[C]-[C] is VALID. (continual)
            drifting as [A]-[B]-[B]-[C]-[A] is INVALID. (back to dist A)
        
    Warning:
        EXTENSION: YES. Datapoints replicated overall. (by dataset_scaling)
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

        VERBOSE is always recommended for sanity check.

        Set DA_dataset_scaling pairing your DA_epoch_locker_num
            when DA_dataset_scaling==DA_epoch_locker_num, #datapoints same as original.

        #TODO to support manual distribution assignment.

    Returns:
        A list of dictionaries contains:
            train, (bool) whether it is training set
            dataset,
            client number,
            epoch locker indicator, (float) to tell the growth time
            epoch locker order: (int) to tell the order of current subset
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert rotation_bank > 0, "The number of rotation patterns must be greater than 0."
    assert color_bank > 0, "The number of color patterns must be greater than 0."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))
    assert 0 <= targeted_class_number <= max_label, "Out of range."
    assert DA_dataset_scaling >= 1, "Invalid downscaling."
    assert DA_epoch_locker_num > 0, "The number of epoch lockers must be greater than 0."
    assert 1 <= DA_max_dist, "Distribution assignment out of range."
    DA_max_dist = min(DA_max_dist, pyx_pattern_bank_num)
    assert torch.unique(train_labels).size(0) == torch.unique(test_labels).size(0), "Original Dataset Fault."
    assert pyx_pattern_bank_num <= math.comb(max_label, targeted_class_number), "pyx_pattern_bank_num out of range."


    # generate pyx bank
    angles = [i * 360 / rotation_bank for i in range(rotation_bank)] if rotation_bank > 1 else [0.0]

    if color_bank == 1:
        colors = ['gray']
    elif color_bank == 2:
        colors = ['red', 'blue']
    elif color_bank == 3:
        colors = ['red', 'blue', 'green']
    else:
        raise ValueError("The number of color patterns must be 1, 2, or 3.")

    px_pattern_bank = [[angle, color] for angle in angles for color in colors]

    pyx_stack = set()
    while len(pyx_stack) < pyx_pattern_bank_num:
        pyx_stack.add(tuple(sorted(np.random.choice(max_label, targeted_class_number, replace=False))))
    pyx_stack = list(pyx_stack)
    random.shuffle(pyx_stack)

    pyx_bank = {i: {'classes': list(pyx_stack.pop()), 
                    'px_pattern': random.choice(px_pattern_bank)}
                for i in range(1, pyx_pattern_bank_num + 1)}

    print("Pyx bank:\n", '\n'.join(f"{key}: {value}" for key, value in pyx_bank.items())) if verbose else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []
    client_Count = 0


    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None
        # training dataset scaling
        original_train_feature = client_data_train['features']
        original_train_label = client_data_train['labels']
        indices = torch.randint(0, original_train_label.shape[0],
                                (int(original_train_label.shape[0] * (DA_dataset_scaling - 1)),))
        sampled_data = original_train_feature[indices]
        sampled_label = original_train_label[indices]

        cur_train_feature = torch.cat((original_train_feature, sampled_data), dim=0)
        cur_train_label = torch.cat((original_train_label, sampled_label), dim=0)
        permuted_indices = torch.randperm(cur_train_label.shape[0])
        cur_train_feature = cur_train_feature[permuted_indices]
        cur_train_label = cur_train_label[permuted_indices]

        cur_test_feature = client_data_test['features']
        cur_test_label = client_data_test['labels']

        # generate drifting
        dist_bank = list(range(1, pyx_pattern_bank_num + 1))
        test_dist = np.random.choice(dist_bank)
        dist_bank.remove(test_dist)
        train_dist = generate_DA_dist(dist_bank,
                                      DA_epoch_locker_num,DA_max_dist,DA_continual_divergence)
        
        lockers = sorted(torch.rand(DA_epoch_locker_num - 1).tolist() + [0.0]) if DA_random_locker \
                else torch.linspace(0, 1, steps=DA_epoch_locker_num + 1)[:-1].tolist()

        print("Train distribution: ", train_dist,
              "\nTest distribution: ", test_dist,
              "\nEpoch lockers: ", lockers,
              "\n") if verbose else None

        # training subsets
        feature_chunks = torch.chunk(cur_train_feature, DA_epoch_locker_num, dim=0)
        label_chunks = torch.chunk(cur_train_label, DA_epoch_locker_num, dim=0)

        # Loop through feature chunks and label chunks
        for i, (feature_chunk, label_chunk) in enumerate(zip(feature_chunks, label_chunks)):

            cur_classes = pyx_bank[train_dist[i]]['classes']
            cur_px_pattern = pyx_bank[train_dist[i]]['px_pattern']

            cur_angle = [float(cur_px_pattern[0]) if label in cur_classes else 0.0 for label in label_chunk]
            cur_color = [cur_px_pattern[1] if label in cur_classes else 'gray' for label in label_chunk]

            # Apply rotation and color transformations
            feature_chunk = rotate_dataset(feature_chunk, cur_angle)
            feature_chunk = color_dataset(feature_chunk, cur_color)

            # Append the cumulative data to rearranged_data
            rearranged_data.append({
                'train': True,
                'features': feature_chunk.detach().cpu().numpy(),
                'labels': label_chunk.detach().cpu().numpy(),
                'client_number': client_Count,
                'epoch_locker_indicator': lockers[i],
                'epoch_locker_order': i,
                'cluster': train_dist[i]
            })

        # testing set
        cur_classes = pyx_bank[test_dist]['classes']
        cur_px_pattern = pyx_bank[test_dist]['px_pattern']

        cur_angle = [float(cur_px_pattern[0]) if label in cur_classes else 0.0 for label in cur_test_label]
        cur_color = [cur_px_pattern[1] if label in cur_classes else 'gray' for label in cur_test_label]

        cur_test_feature = rotate_dataset(cur_test_feature, cur_angle)
        cur_test_feature = color_dataset(cur_test_feature, cur_color)

        rearranged_data.append({
            'train': False,
            'features': cur_test_feature.detach().cpu().numpy(),
            'labels': cur_test_label.detach().cpu().numpy(),
            'client_number': client_Count,
            'epoch_locker_indicator': -1.0,
            'epoch_locker_order': -1,
            'cluster': test_dist
        })

        client_Count += 1

    return rearranged_data
