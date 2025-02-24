import numpy as np
from collections import Counter
from scipy.stats import truncnorm
import itertools
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

def split_feature_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = False,
    rotations: int = 2,
    scaling_rotation_low: float = 0.1,
    scaling_rotation_high: float = 0.1,
    set_color: bool = False,
    colors: int = 2,
    scaling_color_low: float = 0.1,
    scaling_color_high: float = 0.1,
    random_order: bool = True,
    verbose: bool = False
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with ONLY feature skew.
    
    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling_rotation_low (float): The low bound scaling factor of rotation for the softmax distribution.
        scaling_rotation_high (float): The high bound scaling factor of rotation for the softmax distribution.
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [2,3].
        scaling_color_low (float): The low bound scaling factor of color for the softmax distribution.
        scaling_color_high (float): The high bound scaling factor of color for the softmax distribution.
        random_order (bool): Whether to shuffle the order of the rotations and colors.
        verbose (bool): Whether to print the distribution of the assigned features.

    Warning:
        random_order should be identical for both training and testing if not DRIFTING.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_color_high >= scaling_color_low, "High scaling must be larger than low scaling."
    assert scaling_rotation_high >= scaling_rotation_low, "High scaling must be larger than low scaling."

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    # Process train and test splits with rotations if required
    if set_rotation:
        client_Count = 0
        print("Showing rotation distributions..") if verbose else None

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            len_train = len(client_data_train['labels'])
            len_test = len(client_data_test['labels'])
            total_rotations = assigning_rotation_features(
                len_train + len_test, rotations, 
                np.random.uniform(scaling_rotation_low,scaling_rotation_high), random_order
                )
            
            print(f"Client {client_Count}:", dict(Counter(total_rotations))) if verbose else None
            client_Count += 1

            # Split the total_rotations list into train and test
            train_rotations = total_rotations[:len_train]
            test_rotations = total_rotations[len_train:]

            client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)

    if set_color:
        client_Count = 0
        print("Showing color distributions..") if verbose else None

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            len_train = len(client_data_train['labels'])
            len_test = len(client_data_test['labels'])
            total_colors = assigning_color_features(
                len_train + len_test, colors, 
                np.random.uniform(scaling_color_low,scaling_color_high), random_order
                )

            print(f"Client {client_Count}:", dict(Counter(total_colors))) if verbose else None
            client_Count += 1

            # Split the total_colors list into train and test
            train_colors = total_colors[:len_train]
            test_colors = total_colors[len_train:]

            client_data_train['features'] = color_dataset(client_data_train['features'], train_colors)
            client_data_test['features'] = color_dataset(client_data_test['features'], test_colors)

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy(),
            'cluster': -1
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
            
    return rearranged_data
    
def split_label_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor, 
    test_features: torch.Tensor,
    test_labels: torch.Tensor, 
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    verbose: bool = True
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with ONLY label skew.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        scaling_label_low (float): The low bound scaling factor of label for the softmax distribution.
        scaling_label_high (float): The high bound scaling factor of label for the softmax distribution.

    Warning:
        Datasets vary in sensitivity to scaling. Fine-tune the scaling factors for each dataset for optimal results.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_label_high >= scaling_label_low, "High scaling must be larger than low scaling."

    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number

    rearranged_data = []

    remaining_train_features = train_features
    remaining_train_labels = train_labels
    remaining_test_features = test_features
    remaining_test_labels = test_labels

    for i in range(client_number):

        probabilities = calculate_probabilities(remaining_train_labels, np.random.uniform(scaling_label_low,scaling_label_high) * 0.6)

        sub_train_features, sub_train_labels, remaining_train_features, remaining_train_labels = create_sub_dataset(
            remaining_train_features, remaining_train_labels, probabilities, avg_points_per_client_train)
        sub_test_features, sub_test_labels, remaining_test_features, remaining_test_labels = create_sub_dataset(
            remaining_test_features, remaining_test_labels, probabilities, avg_points_per_client_test)
        
        client_data = {
            'train_features': sub_train_features.detach().cpu().numpy(),
            'train_labels': sub_train_labels.detach().cpu().numpy(),
            'test_features': sub_test_features.detach().cpu().numpy(),
            'test_labels': sub_test_labels.detach().cpu().numpy(),
            'cluster': -1
        }        
        rearranged_data.append(client_data)

    return rearranged_data

def split_feature_label_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    set_rotation: bool = False,
    rotations: int = None,
    scaling_rotation_low: float = 0.1,
    scaling_rotation_high: float = 0.1,
    set_color: bool = False,
    colors: int = None,
    scaling_color_low: float = 0.1,
    scaling_color_high: float = 0.1,
    random_order: bool = True,
    verbose: bool = False
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with BOTH feature and label skew.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        scaling_label_low (float): The low bound scaling factor of label for the softmax distribution.
        scaling_label_high (float): The high bound scaling factor of label for the softmax distribution.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling_rotation_low (float): The low bound scaling factor of rotation for the softmax distribution.
        scaling_rotation_high (float): The high bound scaling factor of rotation for the softmax distribution.
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [2,3].
        scaling_color_low (float): The low bound scaling factor of color for the softmax distribution.
        scaling_color_high (float): The high bound scaling factor of color for the softmax distribution.
        random_order (bool): Whether to shuffle the order of the rotations and colors.
        verbose (bool): Whether to print the distribution of the assigned features.

    Warning:
        This should not be used for building concept drift datasets, though unavoidable.
    
    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.

    '''

    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number

    rearranged_data = []

    remaining_train_features = train_features
    remaining_train_labels = train_labels
    remaining_test_features = test_features
    remaining_test_labels = test_labels

    for i in range(client_number):
        print(f"Client {i} feature distributions:") if verbose else None

        probabilities = calculate_probabilities(remaining_train_labels, np.random.uniform(scaling_label_low,scaling_label_high))

        sub_train_features, sub_train_labels, remaining_train_features, remaining_train_labels = create_sub_dataset(
            remaining_train_features, remaining_train_labels, probabilities, avg_points_per_client_train)
        sub_test_features, sub_test_labels, remaining_test_features, remaining_test_labels = create_sub_dataset(
            remaining_test_features, remaining_test_labels, probabilities, avg_points_per_client_test)
        
        if set_rotation:

            len_train = len(sub_train_labels)
            len_test = len(sub_test_labels)
            total_rotations = assigning_rotation_features(
                len_train + len_test, rotations, 
                np.random.uniform(scaling_rotation_low,scaling_rotation_high), random_order
                )
            
            print(dict(Counter(total_rotations))) if verbose else None

            # Split the total_rotations list into train and test
            train_rotations = total_rotations[:len_train]
            test_rotations = total_rotations[len_train:]

            sub_train_features = rotate_dataset(sub_train_features, train_rotations)
            sub_test_features = rotate_dataset(sub_test_features, test_rotations)

        if set_color:

            len_train = len(sub_train_labels)
            len_test = len(sub_test_labels)
            total_colors = assigning_color_features(
                len_train + len_test, colors, 
                np.random.uniform(scaling_color_low,scaling_color_high), random_order
                )
            
            print(dict(Counter(total_colors)),"\n") if verbose else None

            # Split the total_colors list into train and test
            train_colors = total_colors[:len_train]
            test_colors = total_colors[len_train:]

            sub_train_features = color_dataset(sub_train_features, train_colors)
            sub_test_features = color_dataset(sub_test_features, test_colors)

        client_data = {
            'train_features': sub_train_features.detach().cpu().numpy(),
            'train_labels': sub_train_labels.detach().cpu().numpy(),
            'test_features': sub_test_features.detach().cpu().numpy(),
            'test_labels': sub_test_labels.detach().cpu().numpy(),
            'cluster': -1
        }
        rearranged_data.append(client_data)

    return rearranged_data

def split_feature_skew_unbalanced(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = False,
    rotations: int = None,
    scaling_rotation_low: float = 0.1,
    scaling_rotation_high: float = 0.1,
    set_color: bool = False,
    colors: int = None,
    scaling_color_low: float = 0.1,
    scaling_color_high: float = 0.1,
    random_order: bool = True,
    std_dev: float = 0.1,
    permute: bool = True,
    verbose: bool = False
) -> list:
    """
    Splits an overall dataset into a specified number of clusters unbalanced(clients) with feature skew.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling_rotation_low (float): The low bound scaling factor of rotation for the softmax distribution.
        scaling_rotation_high (float): The high bound scaling factor of rotation for the softmax distribution.
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [2,3].
        scaling_color_low (float): The low bound scaling factor of color for the softmax distribution.
        scaling_color_high (float): The high bound scaling factor of color for the softmax distribution.
        random_order (bool): Whether to shuffle the order of the rotations and colors.
        std_dev (float): standard deviation of the normal distribution for the number of samples per client.
        permute (bool): Whether to shuffle the data before splitting.
        verbose (bool): Whether to print the distribution of the assigned features.
        
    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_color_high >= scaling_color_low, "High scaling must be larger than low scaling."
    assert scaling_rotation_high >= scaling_rotation_low, "High scaling must be larger than low scaling."
    assert std_dev > 0, "Standard deviation must be larger than 0."

    # generate basic split unbalanced
    basic_split_data_train = split_unbalanced(train_features, train_labels, client_number, std_dev, permute)
    basic_split_data_test = split_unbalanced(test_features, test_labels, client_number, std_dev, permute)

    if verbose:
        print("Showing unbalanced numbers of data points..") if verbose else None
        client_Count = 0
        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
            print(f"Client {client_Count} Train: {len(client_data_train['labels'])} Test: {len(client_data_test['labels'])}")
            client_Count += 1

    # Process train and test splits with rotations if required
    if set_rotation:
        client_Count = 0
        print("Showing rotation distributions..") if verbose else None

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            len_train = len(client_data_train['labels'])
            len_test = len(client_data_test['labels'])
            total_rotations = assigning_rotation_features(
                len_train + len_test, rotations, 
                np.random.uniform(scaling_rotation_low,scaling_rotation_high), random_order
                )
            
            print(f"Client {client_Count}:", dict(Counter(total_rotations))) if verbose else None
            client_Count += 1

            # Split the total_rotations list into train and test
            train_rotations = total_rotations[:len_train]
            test_rotations = total_rotations[len_train:]

            client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)

    if set_color:
        client_Count = 0
        print("Showing color distributions..") if verbose else None

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            len_train = len(client_data_train['labels'])
            len_test = len(client_data_test['labels'])
            total_colors = assigning_color_features(
                len_train + len_test, colors, 
                np.random.uniform(scaling_color_low,scaling_color_high), random_order
                )
            
            print(f"Client {client_Count}:", dict(Counter(total_colors))) if verbose else None
            client_Count += 1

            # Split the total_colors list into train and test
            train_colors = total_colors[:len_train]
            test_colors = total_colors[len_train:]

            client_data_train['features'] = color_dataset(client_data_train['features'], train_colors)
            client_data_test['features'] = color_dataset(client_data_test['features'], test_colors)

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy(),
            'cluster': -1
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
            
    return rearranged_data

def split_label_skew_unbalanced(
    train_features: torch.Tensor,
    train_labels: torch.Tensor, 
    test_features: torch.Tensor,
    test_labels: torch.Tensor, 
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    std_dev: float = 0.1,
    verbose: bool = True
) -> list:
    """
    Splits an overall dataset into a specified number of clusters unbalanced(clients) with label skew.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        scaling_label_low (float): The low bound scaling factor of label for the softmax distribution.
        scaling_label_high (float): The high bound scaling factor of label for the softmax distribution.
        std_dev (float): standard deviation of the normal distribution for the number of samples per client.
        verbose (bool): Whether to print the number of samples for each client.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert std_dev > 0, "Standard deviation must be larger than 0."
    assert scaling_label_high >= scaling_label_low, "High scaling must be larger than low scaling."

    # Generate different number of samples for each client
    def generate_samples_per_client(features, client_number, std_dev):
        total_samples = len(features)
        percentage = truncnorm.rvs(-0.5/std_dev, 0.5/std_dev, loc=0.5, scale=std_dev, size=client_number)
        normalized_percentage = percentage / np.sum(percentage)
        samples_per_client = (normalized_percentage * total_samples).astype(int)
        return samples_per_client

    train_samples_per_client = generate_samples_per_client(train_features, client_number, std_dev)
    test_samples_per_client = generate_samples_per_client(test_features, client_number, std_dev)

    rearranged_data = []

    remaining_train_features = train_features
    remaining_train_labels = train_labels
    remaining_test_features = test_features
    remaining_test_labels = test_labels

    for i in range(client_number):
        print(f'Client {i+1} - Train: {train_samples_per_client[i]} Test: {test_samples_per_client[i]}') if verbose else None

        # if i == client_number - 1:
        #     client_data = {
        #         'train_features': remaining_train_features,
        #         'train_labels': remaining_train_labels,
        #         'test_features': remaining_test_features,
        #         'test_labels': remaining_test_labels
        #     } 
        #     rearranged_data.append(client_data)
        #     break

        probabilities = calculate_probabilities(remaining_train_labels, np.random.uniform(scaling_label_low,scaling_label_high))

        sub_train_features, sub_train_labels, remaining_train_features, remaining_train_labels = create_sub_dataset(
            remaining_train_features, remaining_train_labels, probabilities, train_samples_per_client[i])
        sub_test_features, sub_test_labels, remaining_test_features, remaining_test_labels = create_sub_dataset(
            remaining_test_features, remaining_test_labels, probabilities, test_samples_per_client[i])
        
        client_data = {
            'train_features': sub_train_features.detach().cpu().numpy(),
            'train_labels': sub_train_labels.detach().cpu().numpy(),
            'test_features': sub_test_features.detach().cpu().numpy(),
            'test_labels': sub_test_labels.detach().cpu().numpy()
        }        
        rearranged_data.append(client_data)

    return rearranged_data

def split_label_condition_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    random_mode: bool = True,
    mixing_label_number: int = 3,
    mixing_label_list: list = None,
    scaling_label_low: float = 0.0,
    scaling_label_high: float = 0.0,
    verbose: bool = False
) -> list:
    """
    P(y|x) differs across clients by label swapping.

    Random mode: randomly choose which labels are in the swapping pool. (#mixing_label_number)
    Non-random mode: a list of labels are provided to be swapped.

    A scaling factor is randomly generated. (scaling_label_low to scaling_label_high)
    When 1, dirichlet shuffling, when 0, no shuffling.

    Warning:
        The re-mapping choices of labels are growing with the swapping pool.
        E.g. When the swapping pool has [1,2,3]. Label '3' could be swapped with both '1' or '2'.

        USE Non-random mode for EMNIST, which is the only dataset doesn't start label from 0.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        random_mode (bool): Random mode.
        mixing_label_number (int): The number of labels to swap in Random mode.
        mixing_label_list (list): A list of labels to swap in Non-random mode.
        scaling_label_low (float): The low bound scaling factor of label skewing.
        scaling_label_high (float): The high bound scaling factor of label skewing.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 0 <= scaling_label_low <= scaling_label_high <= 1, "Scaling factor must be between 0 and 1."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert mixing_label_number > 0, "The number of labels to swap must be larger than 0."
        mixing_label_list = np.random.choice(range(0, max_label), mixing_label_number,replace=False).tolist()
    else:
        assert mixing_label_list is not None, "The list of labels to swap must be provided."
        assert len(mixing_label_list) == len(set(mixing_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in mixing_label_list), "Label out of range."
    
    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []

    print("Showing the label mapping for each client..") if verbose else None
    n_clusters = 0
    list_label_maps = []
    dict_label_maps = {}

    for i in range(client_number):

        scaling_label = np.random.uniform(scaling_label_low, scaling_label_high)
        print("Re-maping possibility for this client is: ", scaling_label) if verbose else None

        # Mapping from original label to the permuted label
        permuted_label_list = mixing_label_list.copy()
        np.random.shuffle(permuted_label_list)
        label_map = {original: permuted for original, permuted in zip(mixing_label_list, permuted_label_list)}

        print(f'Client {i+1} - Label Mapping: {label_map}') if verbose else None

        # Check if label_map is already in the list, otherwise append
        if label_map not in list_label_maps:
            dict_label_maps[i] = n_clusters
            list_label_maps.append(label_map)
            n_clusters += 1
        else: # if exists, get the index
            dict_label_maps[i] = list_label_maps.index(label_map)

        new_train_labels = basic_split_data_train[i]['labels'].clone()
        new_test_labels = basic_split_data_test[i]['labels'].clone()
        
        for original, permuted in label_map.items():
            # Replace labels based on the scaling_label probability
            train_mask = (basic_split_data_train[i]['labels'] == original)
            test_mask = (basic_split_data_test[i]['labels'] == original)
            
            random_values_train = torch.rand(train_mask.sum().item())
            random_values_test = torch.rand(test_mask.sum().item())
            
            new_train_labels[train_mask] = torch.where(random_values_train <= scaling_label, permuted, original)
            new_test_labels[test_mask] = torch.where(random_values_test <= scaling_label, permuted, original)

        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': new_train_labels.detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': new_test_labels.detach().cpu().numpy(),
            'cluster': dict_label_maps[i]
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
    
    return rearranged_data

def split_label_condition_skew_unbalanced(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    random_mode: bool = True,
    mixing_label_number: int = None,
    mixing_label_list: list = None,
    scaling_label_low: float = 0.0,
    scaling_label_high: float = 0.0,
    std_dev: float = 0.1,
    permute: bool = True,
    verbose: bool = False
) -> list:
    """
    P(y|x) differs across clients by label swapping.

    Random mode: randomly choose which labels are in the swapping pool. (#mixing_label_number)
    Non-random mode: a list of labels are provided to be swapped.

    Warning:
        The re-mapping possibility of labels are growing with the swapping pool.
        E.g. When the swapping pool has [1,2,3]. Label '3' could be swapped with both '1' or '2'.

        USE Non-random mode for EMNIST, which is the only dataset doesn't start label from 0.
    
    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        random_mode (bool): Random mode.
        mixing_label_number (int): The number of labels to swap in Random mode.
        mixing_label_list (list): A list of labels to swap in Non-random mode.
        scaling_label_low (float): The low bound scaling factor of label skewing.
        scaling_label_high (float): The high bound scaling factor of label skewing.
        std_dev (float): standard deviation of the normal distribution for the number of samples per client.
        permute (bool): Whether to shuffle the data before splitting.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 0 <= scaling_label_low <= scaling_label_high <= 1, "Scaling factor must be between 0 and 1."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert mixing_label_number > 0, "The number of labels to swap must be larger than 0."
        mixing_label_list = np.random.choice(range(0, max_label), mixing_label_number,replace=False).tolist()
    else:
        assert mixing_label_list is not None, "The list of labels to swap must be provided."
        assert len(mixing_label_list) == len(set(mixing_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in mixing_label_list), "Label out of range."
    assert std_dev > 0, "Standard deviation must be larger than 0."

    # generate basic split unbalanced
    basic_split_data_train = split_unbalanced(train_features, train_labels, client_number, std_dev, permute)
    basic_split_data_test = split_unbalanced(test_features, test_labels, client_number, std_dev, permute)

    rearranged_data = []

    print("Showing the label mapping for each client..") if verbose else None

    for i in range(client_number):

        scaling_label = np.random.uniform(scaling_label_low, scaling_label_high)
        print("Re-maping possibility for this client is: ", scaling_label) if verbose else None

        # Mapping from original label to the permuted label
        permuted_label_list = mixing_label_list.copy()
        np.random.shuffle(permuted_label_list)
        label_map = {original: permuted for original, permuted in zip(mixing_label_list, permuted_label_list)}

        print(f'Client {i+1} - Label Mapping: {label_map}') if verbose else None

        new_train_labels = basic_split_data_train[i]['labels'].clone()
        new_test_labels = basic_split_data_test[i]['labels'].clone()
        
        for original, permuted in label_map.items():
            # Replace labels based on the scaling_label probability
            train_mask = (basic_split_data_train[i]['labels'] == original)
            test_mask = (basic_split_data_test[i]['labels'] == original)
            
            random_values_train = torch.rand(train_mask.sum().item())
            random_values_test = torch.rand(test_mask.sum().item())
            
            new_train_labels[train_mask] = torch.where(random_values_train <= scaling_label, permuted, original)
            new_test_labels[test_mask] = torch.where(random_values_test <= scaling_label, permuted, original)

        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': new_train_labels.detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': new_test_labels.detach().cpu().numpy(),
            'cluster': -1
        }

        rearranged_data.append(client_data)
    
    return rearranged_data

def split_feature_condition_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = False,
    rotations: int = 2,
    set_color: bool = False,
    colors: int = 3,
    random_mode: bool = True,
    rotated_label_number: int = 2,
    colored_label_number: int = 2,
    rotated_label_list: list = None,
    colored_label_list: list = None,
    verbose: bool = False
) -> list:
    """
    P(x|y) differs across clients by targeted rotation/coloring.

    Random mode: randomly choose which labels are to be rotated/colored. (#rotated_label_number/#colored_label_number)
    Non-random mode: a list of labels are provided to be rotated/colored.

    Scaling is not yet supported. Didn't see much point of that.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [1,2,3].
        random_mode (bool): Random mode.
        rotated_label_number (int): The number of labels to rotate in Random mode.
        colored_label_number (int): The number of labels to color in Random mode.
        rotated_label_list (list): A list of labels to rotate in Non-random mode.
        colored_label_list (list): A list of labels to color in Non-random mode.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 1 <= colors <= 3, "The number of colors must be 1, 2, or 3."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert rotated_label_number is not None or colored_label_number is not None, "The number of labels to rotate/color must be provided."
        rotated_label_list = np.random.choice(range(0, max_label), rotated_label_number,replace=False).tolist()
        colored_label_list = np.random.choice(range(0, max_label), colored_label_number,replace=False).tolist()
    else:
        assert rotated_label_list is not None or colored_label_list is not None, "The list of labels to rotate/color must be provided."
        assert len(rotated_label_list) == len(set(rotated_label_list)), "Repeated list."
        assert len(colored_label_list) == len(set(colored_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in rotated_label_list), "Label out of range."
        assert all(0 <= label <= max_label for label in colored_label_list), "Label out of range."
    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    n_clusters = 0
    list_r_maps = []
    dict_r_maps = {}

    # Example usage within the split_label_condition_skew function
    if set_rotation:
        client_Count = 0
        print("Showing rotated labels for each client..") if verbose else None

        angles = [i * 360 / rotations for i in range(rotations)]

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            rotation_mapping = {label: np.random.choice(angles) if label in rotated_label_list else 0.0
                                for label in np.arange(0, max_label+1).tolist()}
            
            print(f'Client {client_Count} rotation mapping: {rotation_mapping}') if verbose else None

            # Check if label_map is already in the list, otherwise append
            if rotation_mapping not in list_r_maps:
                dict_r_maps[client_Count] = n_clusters
                list_r_maps.append(rotation_mapping)
                n_clusters += 1
            else: # if exists, get the index
                dict_r_maps[client_Count] = list_r_maps.index(rotation_mapping)

            client_Count += 1

            train_rotations = [rotation_mapping[label.item()] for label in client_data_train['labels']]
            test_rotations = [rotation_mapping[label.item()] for label in client_data_test['labels']]

            client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)

    if set_color:
        client_Count = 0
        print("Showing colored labels for each client..") if verbose else None
        letters = ['red', 'blue', 'green'][:colors] if colors != 1 else ['gray']

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            color_mapping = {label: np.random.choice(letters) if label in colored_label_list else "gray"
                                for label in np.arange(0, max_label+1).tolist()}
            
            print(f'Client {client_Count} color mapping: {color_mapping}') if verbose else None
            client_Count += 1

            train_colors = [color_mapping[label.item()] for label in client_data_train['labels']]
            test_colors = [color_mapping[label.item()] for label in client_data_test['labels']]

            client_data_train['features'] = color_dataset(client_data_train['features'], train_colors)
            client_data_test['features'] = color_dataset(client_data_test['features'], test_colors)

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy(),
            'cluster': dict_r_maps[i]
        }

        rearranged_data.append(client_data)
            
    return rearranged_data

def split_feature_condition_skew_unbalanced(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = False,
    rotations: int = 2,
    set_color: bool = False,
    colors: int = 3,
    random_mode: bool = True,
    rotated_label_number: int = 2,
    colored_label_number: int = 2,
    rotated_label_list: list = None,
    colored_label_list: list = None,
    std_dev: float = 0.1,
    permute: bool = True,
    verbose: bool = False
) -> list:
    """
    P(x|y) differs across clients by targeted rotation/coloring.

    Random mode: randomly choose which labels are to be rotated/colored. (#rotated_label_number/#colored_label_number)
    Non-random mode: a list of labels are provided to be rotated/colored.

    Scaling is not yet supported. Didn't see much point of that.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [1,2,3].
        random_mode (bool): Random mode.
        rotated_label_number (int): The number of labels to rotate in Random mode.
        colored_label_number (int): The number of labels to color in Random mode.
        rotated_label_list (list): A list of labels to rotate in Non-random mode.
        colored_label_list (list): A list of labels to color in Non-random mode.
        std_dev (float): standard deviation of the normal distribution for the number of samples per client.
        permute (bool): Whether to shuffle the data before splitting.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert rotations > 1, "Must have at least 2 rotations. Otherwise turn it off."
    assert colors == 1 or colors == 2 or colors == 3, "The number of colors must be 1, 2, or 3."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert rotated_label_number is not None or colored_label_number is not None, "The number of labels to rotate/color must be provided."
        rotated_label_list = np.random.choice(range(0, max_label), rotated_label_number,replace=False).tolist()
        colored_label_list = np.random.choice(range(0, max_label), colored_label_number,replace=False).tolist()
    else:
        assert rotated_label_list is not None or colored_label_list is not None, "The list of labels to rotate/color must be provided."
        assert len(rotated_label_list) == len(set(rotated_label_list)), "Repeated list."
        assert len(colored_label_list) == len(set(colored_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in rotated_label_list), "Label out of range."
        assert all(0 <= label <= max_label for label in colored_label_list), "Label out of range."

    # generate basic split unbalanced
    basic_split_data_train = split_unbalanced(train_features, train_labels, client_number, std_dev, permute)
    basic_split_data_test = split_unbalanced(test_features, test_labels, client_number, std_dev, permute)

    # Example usage within the split_label_condition_skew function
    if set_rotation:
        client_Count = 0
        print("Showing rotated labels for each client..") if verbose else None

        angles = [i * 360 / rotations for i in range(rotations)]

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            rotation_mapping = {label: np.random.choice(angles) if label in rotated_label_list else 0.0
                                for label in np.arange(0, max_label+1).tolist()}
            
            print(f'Client {client_Count} rotation mapping: {rotation_mapping}') if verbose else None
            client_Count += 1

            train_rotations = [rotation_mapping[label.item()] for label in client_data_train['labels']]
            test_rotations = [rotation_mapping[label.item()] for label in client_data_test['labels']]

            client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)

    if set_color:
        client_Count = 0
        print("Showing colored labels for each client..") if verbose else None

        letters = ['red', 'blue', 'green'][:colors]

        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            color_mapping = {label: np.random.choice(letters) if label in colored_label_list else "gray"
                                for label in np.arange(0, max_label+1).tolist()}
            
            print(f'Client {client_Count} color mapping: {color_mapping}') if verbose else None
            client_Count += 1

            train_colors = [color_mapping[label.item()] for label in client_data_train['labels']]
            test_colors = [color_mapping[label.item()] for label in client_data_test['labels']]

            client_data_train['features'] = color_dataset(client_data_train['features'], train_colors)
            client_data_test['features'] = color_dataset(client_data_test['features'], test_colors)

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy(),
            'cluster': -1
        }

        rearranged_data.append(client_data)
            
    return rearranged_data

def split_label_condition_skew_with_label_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor, 
    test_features: torch.Tensor,
    test_labels: torch.Tensor, 
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    random_mode: bool = True,
    mixing_label_number: int = 2,
    mixing_label_list: list = None,
    scaling_swapping_low: float = 0.0,
    scaling_swapping_high: float = 0.0,
    verbose: bool = False
) -> list:
    '''
    P(y|x) differs across clients by label swapping while clients already label skewed.

    Random mode: randomly choose which labels are in the swapping pool. (#mixing_label_number)
    Non-random mode: a list of labels are provided to be swapped.

    A scaling factor is randomly generated. When 1, dirichlet shuffling, when 0, no shuffling.

    Warning:
        The re-mapping possibility of labels are growing with the swapping pool.
        E.g. When the swapping pool has [1,2,3]. Label '3' could be swapped with both '1' or '2'.

        USE Non-random mode for EMNIST, which is the only dataset doesn't start label from 0.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        scaling_label_low (float): The low bound scaling factor of label for the softmax distribution.
        scaling_label_high (float): The high bound scaling factor of label for the softmax distribution.
        random_mode (bool): Random mode.
        mixing_label_number (int): The number of labels to swap in Random mode.
        mixing_label_list (list): A list of labels to swap in Non-random mode.
        scaling_swapping_low (float): The low bound scaling factor of label skewing.
        scaling_swapping_high (float): The high bound scaling factor of label skewing.
        verbose (bool): Whether to print the number of samples for each client.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_label_high >= scaling_label_low, "High scaling must be larger than low scaling."
    assert scaling_swapping_high >= scaling_swapping_low, "High scaling must be larger than low scaling."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert mixing_label_number > 0, "The number of labels to swap must be larger than 0."
        mixing_label_list = np.random.choice(range(0, max_label), mixing_label_number,replace=False).tolist()
    else:
        assert mixing_label_list is not None, "The list of labels to swap must be provided."
        assert len(mixing_label_list) == len(set(mixing_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in mixing_label_list), "Label out of range."

    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number

    rearranged_data = []

    remaining_train_features = train_features
    remaining_train_labels = train_labels
    remaining_test_features = test_features
    remaining_test_labels = test_labels

    print("Showing label mapping for each client..") if verbose else None

    for i in range(client_number):

        probabilities = calculate_probabilities(remaining_train_labels, np.random.uniform(scaling_label_low,scaling_label_high))

        sub_train_features, sub_train_labels, remaining_train_features, remaining_train_labels = create_sub_dataset(
            remaining_train_features, remaining_train_labels, probabilities, avg_points_per_client_train)
        sub_test_features, sub_test_labels, remaining_test_features, remaining_test_labels = create_sub_dataset(
            remaining_test_features, remaining_test_labels, probabilities, avg_points_per_client_test)
        
        # feature condition skew

        scaling_swapping = np.random.uniform(scaling_swapping_low, scaling_swapping_high)

        # Mapping from original label to the permuted label
        permuted_label_list = mixing_label_list.copy()
        np.random.shuffle(permuted_label_list)
        label_map = {original: permuted for original, permuted in zip(mixing_label_list, permuted_label_list)}
        print(f'Client {i} - Label Mapping: {label_map}') if verbose else None
        print(f'Client {i} remapping probability: {scaling_swapping}\n') if verbose else None

        new_train_labels = sub_train_labels.clone()
        new_test_labels = sub_test_labels.clone()

        for original, permuted in label_map.items():
            # Replace labels based on the scaling_label probability
            train_mask = (sub_train_labels == original)
            test_mask = (sub_test_labels == original)
            
            random_values_train = torch.rand(train_mask.sum().item())
            random_values_test = torch.rand(test_mask.sum().item())
            
            new_train_labels[train_mask] = torch.where(random_values_train <= scaling_swapping, permuted, original)
            new_test_labels[test_mask] = torch.where(random_values_test <= scaling_swapping, permuted, original)

        client_data = {
            'train_features': sub_train_features.detach().cpu().numpy(),
            'train_labels': new_train_labels.detach().cpu().numpy(),
            'test_features': sub_test_features.detach().cpu().numpy(),
            'test_labels': new_test_labels.detach().cpu().numpy(),
            'cluster': -1
        }        
        rearranged_data.append(client_data)

    return rearranged_data

def split_feature_condition_skew_with_label_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor, 
    test_features: torch.Tensor,
    test_labels: torch.Tensor, 
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    set_rotation: bool = False,
    rotations: int = 2,
    set_color: bool = False,
    colors: int = 3,
    random_mode: bool = True,
    rotated_label_number: int = 2,
    colored_label_number: int = 2,
    rotated_label_list: list = None,
    colored_label_list: list = None,
    verbose: bool = False
) -> list:
    '''
    P(x|y) differs across clients by targeted rotation/coloring while clients already label skewed.

    Random mode: randomly choose which labels are to be rotated/colored. (#rotated_label_number/#colored_label_number)
    Non-random mode: a list of labels are provided to be rotated/colored.

    Scaling is not yet supported. Didn't see much point of that.
    
    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        scaling_label_low (float): The low bound scaling factor of label for the softmax distribution.
        scaling_label_high (float): The high bound scaling factor of label for the softmax distribution.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [1,2,3].
        random_mode (bool): Random mode.
        rotated_label_number (int): The number of labels to rotate in Random mode.
        colored_label_number (int): The number of labels to color in Random mode.
        rotated_label_list (list): A list of labels to rotate in Non-random mode.
        colored_label_list (list): A list of labels to color in Non-random mode.
        verbose (bool): Whether to print the number of samples for each client.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_label_high >= scaling_label_low, "High scaling must be larger than low scaling."
    assert rotations > 1, "Must have at least 2 rotations. Otherwise turn it off."
    assert colors == 1 or colors == 2 or colors == 3, "The number of colors must be 1, 2, or 3."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert rotated_label_number is not None or colored_label_number is not None, "The number of labels to rotate/color must be provided."
        rotated_label_list = np.random.choice(range(0, max_label), rotated_label_number,replace=False).tolist()
        colored_label_list = np.random.choice(range(0, max_label), colored_label_number,replace=False).tolist()
    else:
        assert rotated_label_list is not None or colored_label_list is not None, "The list of labels to rotate/color must be provided."
        assert len(rotated_label_list) == len(set(rotated_label_list)), "Repeated list."
        assert len(colored_label_list) == len(set(colored_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in rotated_label_list), "Label out of range."
        assert all(0 <= label <= max_label for label in colored_label_list), "Label out of range."

    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number

    rearranged_data = []

    remaining_train_features = train_features
    remaining_train_labels = train_labels
    remaining_test_features = test_features
    remaining_test_labels = test_labels

    letters = ['red', 'blue', 'green'][:colors]

    for i in range(client_number):
        print(f'Showing mappings for Client {i}') if verbose else None

        probabilities = calculate_probabilities(remaining_train_labels, np.random.uniform(scaling_label_low,scaling_label_high))

        sub_train_features, sub_train_labels, remaining_train_features, remaining_train_labels = create_sub_dataset(
            remaining_train_features, remaining_train_labels, probabilities, avg_points_per_client_train)
        sub_test_features, sub_test_labels, remaining_test_features, remaining_test_labels = create_sub_dataset(
            remaining_test_features, remaining_test_labels, probabilities, avg_points_per_client_test)
        

        if set_rotation:
            angles = [i * 360 / rotations for i in range(rotations)]

            rotation_mapping = {label: np.random.choice(angles) if label in rotated_label_list else 0.0
                                for label in np.arange(0, max_label+1).tolist()}
            
            print(f'Rotation Mapping: {rotation_mapping}') if verbose else None

            train_rotations = [rotation_mapping[label.item()] for label in sub_train_labels]
            test_rotations = [rotation_mapping[label.item()] for label in sub_test_labels]

            sub_train_features = rotate_dataset(sub_train_features, train_rotations)
            sub_test_features = rotate_dataset(sub_test_features, test_rotations)
        
        if set_color:
            color_mapping = {label: np.random.choice(letters) if label in colored_label_list else "gray"
                                for label in np.arange(0, max_label+1).tolist()}
            
            print(f'Color Mapping: {color_mapping}\n') if verbose else None

            train_colors = [color_mapping[label.item()] for label in sub_train_labels]
            test_colors = [color_mapping[label.item()] for label in sub_test_labels]

            sub_train_features = color_dataset(sub_train_features, train_colors)
            sub_test_features = color_dataset(sub_test_features, test_colors)
        
        client_data = {
            'train_features': sub_train_features.detach().cpu().numpy(),
            'train_labels': sub_train_labels.detach().cpu().numpy(),
            'test_features': sub_test_features.detach().cpu().numpy(),
            'test_labels': sub_test_labels.detach().cpu().numpy(),
            'cluster': -1
        }        
        rearranged_data.append(client_data)

    return rearranged_data

def split_feature_skew_strict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = True,
    rotations: int = 2,
    set_color: bool = True,
    colors: int = 2,
    verbose: bool = True
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with ONLY feature skew.
    
    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling_rotation_low (float): The low bound scaling factor of rotation for the softmax distribution.
        scaling_rotation_high (float): The high bound scaling factor of rotation for the softmax distribution.
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [2,3].
        scaling_color_low (float): The low bound scaling factor of color for the softmax distribution.
        scaling_color_high (float): The high bound scaling factor of color for the softmax distribution.
        verbose (bool): Whether to print the distribution of the assigned features.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert colors in [1, 2, 3], "The number of colors must be 1, 2 or 3."

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    n_rotations = [i * 360 / rotations for i in range(rotations)]
    if colors == 1:
        n_colors = ['gray']
    elif colors == 2:
        n_colors = ['red', 'blue']
    else:
        n_colors = ['red', 'blue', 'green']
    pattern_bank = {i: [r, c] for i, (r, c) in enumerate([(r, c) for r in n_rotations for c in n_colors])}
    client_clusters = np.random.randint(0, len(pattern_bank), size=client_number).tolist()

    print("Pattern bank:\n", '\n'.join(f"{key}: {value}" for key, value in pattern_bank.items())) if verbose else None
    print("Client clusters:", client_clusters) if verbose else None

    # Process train and test splits with rotations if required
    client_Count = 0
    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

        len_train, len_test = len(client_data_train['labels']), len(client_data_test['labels'])
        cur_cluster = client_clusters[client_Count]

        # Handle rotation if required
        if set_rotation:
            client_data_train['features'] = rotate_dataset(client_data_train['features'], [pattern_bank[cur_cluster][0]] * len_train)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], [pattern_bank[cur_cluster][0]] * len_test)

        if set_color:
            client_data_train['features'] = color_dataset(client_data_train['features'], [pattern_bank[cur_cluster][1]] * len_train)
            client_data_test['features'] = color_dataset(client_data_test['features'], [pattern_bank[cur_cluster][1]] * len_test)

        client_Count += 1

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy(),
            'cluster': client_clusters[i]
        }
        rearranged_data.append(client_data)
            
    return rearranged_data

def split_label_skew_strict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor, 
    test_features: torch.Tensor,
    test_labels: torch.Tensor, 
    client_number: int = 10,
    client_n_class: int = 2,
    py_bank: int = 3,
    verbose: bool = True,
    extend: bool = True
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with ONLY label skew.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        client_n_class (int): The number of classes for each client.
        py_bank (int): The number of possible label distributions.

    Warning:
        Datasets vary in sensitivity to scaling. Fine-tune the scaling factors for each dataset for optimal results.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    label_num = torch.unique(train_labels).size(0)
    assert 1 <= client_n_class <= label_num, "Invalid number of classes per set."

    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number

    py_class_bank = {i: sorted(np.random.choice(label_num, client_n_class, replace=False).tolist())
                   for i in range(1, py_bank + 1)}
    print("Py bank:\n", '\n'.join(f"{key}: {value}" for key, value in py_class_bank.items())) if verbose else None
    
    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []
    client_Count = 0

    extend_factor = label_num / client_n_class if extend else 1

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

        cur_train_feature = client_data_train['features']
        cur_train_label = client_data_train['labels']
        cur_test_feature = client_data_test['features']
        cur_test_label = client_data_test['labels']

        # generate drifting
        dist = int(np.random.choice(list(range(1, py_bank + 1))))
        print(f"Client: {client_Count}, Distribution: {dist} ") if verbose else None

        # train set
        locker_classes_tensor = torch.tensor(py_class_bank[dist])
        mask = torch.isin(cur_train_label, locker_classes_tensor)
        filtered_train_feature, filtered_train_label  = cur_train_feature[mask], cur_train_label[mask]

        # testing set
        locker_classes_tensor = torch.tensor(py_class_bank[dist])
        mask = torch.isin(cur_test_label, locker_classes_tensor)
        filtered_test_feature, filtered_test_label = cur_test_feature[mask], cur_test_label[mask]

        extend_size_train = int(filtered_train_feature.shape[0] * (extend_factor - 1))
        extend_size_test = int(filtered_test_feature.shape[0] * (extend_factor - 1))

        # Repeat random indices to extend the dataset
        repeat_train_indices = np.random.choice(filtered_train_feature.shape[0], extend_size_train, replace=True)
        repeat_test_indices = np.random.choice(filtered_test_feature.shape[0], extend_size_test, replace=True)

        # Extend training data
        extended_train_features = np.concatenate([filtered_train_feature, filtered_train_feature[repeat_train_indices]], axis=0)
        extended_train_labels = np.concatenate([filtered_train_label, filtered_train_label[repeat_train_indices]], axis=0)
        
        # Extend test data
        extended_test_features = np.concatenate([filtered_test_feature, filtered_test_feature[repeat_test_indices]], axis=0)
        extended_test_labels = np.concatenate([filtered_test_label, filtered_test_label[repeat_test_indices]], axis=0)

        rearranged_data.append({
            'train_features': torch.from_numpy(extended_train_features).detach().cpu().numpy(),
            'train_labels': torch.from_numpy(extended_train_labels).detach().cpu().numpy(),
            'test_features': torch.from_numpy(extended_test_features).detach().cpu().numpy(),
            'test_labels': torch.from_numpy(extended_test_labels).detach().cpu().numpy(),
            'cluster': dist
        })

        client_Count += 1

    return rearranged_data

def split_label_condition_skew_strict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    random_mode: bool = True,
    mixing_label_number: int = 3,
    mixing_label_list: list = None,
    verbose: bool = False
) -> list:
    """
    P(y|x) differs across clients by label swapping.

    Random mode: randomly choose which labels are in the swapping pool. (#mixing_label_number)
    Non-random mode: a list of labels are provided to be swapped.

    A scaling factor is randomly generated. (scaling_label_low to scaling_label_high)
    When 1, dirichlet shuffling, when 0, no shuffling.

    Warning:
        The re-mapping choices of labels are growing with the swapping pool.
        E.g. When the swapping pool has [1,2,3]. Label '3' could be swapped with both '1' or '2'.

        USE Non-random mode for EMNIST, which is the only dataset doesn't start label from 0.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        random_mode (bool): Random mode.
        mixing_label_number (int): The number of labels to swap in Random mode.
        mixing_label_list (list): A list of labels to swap in Non-random mode.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert mixing_label_number > 0, "The number of labels to swap must be larger than 0."
        mixing_label_list = np.random.choice(range(0, max_label), mixing_label_number,replace=False).tolist()
    else:
        assert mixing_label_list is not None, "The list of labels to swap must be provided."
        assert len(mixing_label_list) == len(set(mixing_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in mixing_label_list), "Label out of range."
    
    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    rearranged_data = []

    print("Showing the label mapping for each client..") if verbose else None

    # Generate clusters
    all_permutations = itertools.permutations(mixing_label_list)
    all_label_maps = []
    for permuted_label_list in all_permutations:
        label_map = {original: permuted for original, permuted in zip(mixing_label_list, permuted_label_list)}
        all_label_maps.append(label_map)
    print("All label maps:\n", '\n'.join(f"{key}: {value}" for key, value in enumerate(all_label_maps))) if verbose else None

    for i in range(client_number):
        cur_cluster = int(np.random.randint(0, len(all_label_maps)))
        label_map = all_label_maps[cur_cluster]

        print(f'Client {i} - Cluster {cur_cluster}') if verbose else None

        new_train_labels = basic_split_data_train[i]['labels'].clone()
        new_test_labels = basic_split_data_test[i]['labels'].clone()
        
        for original, permuted in label_map.items():
            # Create masks to identify where labels are equal to the original label
            train_mask = (basic_split_data_train[i]['labels'] == original)
            test_mask = (basic_split_data_test[i]['labels'] == original)
            
            # Directly assign the permuted label where the mask is true
            new_train_labels[train_mask] = permuted
            new_test_labels[test_mask] = permuted

        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': new_train_labels.detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': new_test_labels.detach().cpu().numpy(),
            'cluster': cur_cluster
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
    
    return rearranged_data

def split_feature_condition_skew_strict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = False,
    rotations: int = 2,
    set_color: bool = True,
    colors: int = 3,
    random_mode: bool = True,
    rotated_label_number: int = 2,
    colored_label_number: int = 2,
    rotated_label_list: list = None,
    colored_label_list: list = None,
    verbose: bool = False
) -> list:
    """
    P(x|y) differs across clients by targeted rotation/coloring.

    Random mode: randomly choose which labels are to be rotated/colored. (#rotated_label_number/#colored_label_number)
    Non-random mode: a list of labels are provided to be rotated/colored.

    Scaling is not yet supported. Didn't see much point of that.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [1,2,3].
        random_mode (bool): Random mode.
        rotated_label_number (int): The number of labels to rotate in Random mode.
        colored_label_number (int): The number of labels to color in Random mode.
        rotated_label_list (list): A list of labels to rotate in Non-random mode.
        colored_label_list (list): A list of labels to color in Non-random mode.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert rotations > 1, "Must have at least 2 rotations. Otherwise turn it off."
    assert 1 <= colors <= 3, "The number of colors must be 1, 2, or 3."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))

    if random_mode:
        assert rotated_label_number is not None or colored_label_number is not None, "The number of labels to rotate/color must be provided."
        rotated_label_list = np.random.choice(range(0, max_label), rotated_label_number,replace=False).tolist()
        colored_label_list = np.random.choice(range(0, max_label), colored_label_number,replace=False).tolist()
    else:
        assert rotated_label_list is not None or colored_label_list is not None, "The list of labels to rotate/color must be provided."
        assert len(rotated_label_list) == len(set(rotated_label_list)), "Repeated list."
        assert len(colored_label_list) == len(set(colored_label_list)), "Repeated list."
        assert all(0 <= label <= max_label for label in rotated_label_list), "Label out of range."
        assert all(0 <= label <= max_label for label in colored_label_list), "Label out of range."
    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    angles = [i * 360 / rotations for i in range(rotations)] if set_rotation else [0.0]
    letters = ['red', 'blue', 'green'][:colors] if set_color else ['gray']
    permuted_list = list(itertools.product(angles, letters))
    print("Clusters:\n", '\n'.join(f"{key}: {value}" for key, value in enumerate(permuted_list))) if verbose else None

    client_clusters = np.random.randint(0, len(permuted_list), size=client_number).tolist()
    print("Client clusters:", client_clusters) if verbose else None

    client_Count = 0
    
    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        
        cur_cluster = client_clusters[client_Count]

        # Rotation handling
        if set_rotation:
            rotation_mapping = {
                label: permuted_list[cur_cluster][0] if label in rotated_label_list else 0.0
                for label in np.arange(0, max_label+1).tolist()
            }
            print(f'Client {client_Count} rotation mapping: {rotation_mapping}') if verbose else None

            train_rotations = [rotation_mapping[label.item()] for label in client_data_train['labels']]
            test_rotations = [rotation_mapping[label.item()] for label in client_data_test['labels']]

            client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)

        # Color handling
        if set_color:
            color_mapping = {
                label: permuted_list[cur_cluster][1] if label in colored_label_list else "gray"
                for label in np.arange(0, max_label+1).tolist()
            }
            print(f'Client {client_Count} color mapping: {color_mapping}') if verbose else None

            train_colors = [color_mapping[label.item()] for label in client_data_train['labels']]
            test_colors = [color_mapping[label.item()] for label in client_data_test['labels']]

            client_data_train['features'] = color_dataset(client_data_train['features'], train_colors)
            client_data_test['features'] = color_dataset(client_data_test['features'], test_colors)

        client_Count += 1

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy(),
            'cluster': client_clusters[i]
        }

        rearranged_data.append(client_data)
            
    return rearranged_data
