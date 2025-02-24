import numpy as np
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

def split_trND_teDR_Px(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    rotation_bank: int = 2,
    color_bank: int = 3,
    scaling_low: float = 0.5,
    scaling_high: float = 0.5,
    reverse_test: bool = True,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A (large in size)
        Testing: B (unseen)
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
        scaling_low (float): The lower bound of scaling.
        scaling_high (float): The upper bound of scaling.
        reverse_test (bool): Testing and training use reverse patterns. (used to create strong unseen level)
        verbose (bool): Whether to print the distribution information.

    Description:
        A pattern bank will be created based on #rotation and #color.
        Each DATAPOINT applies one of the patterns in the bank.
        Each sub-dataset choose any distribution of the patterns.
        
    Warning:
        EXTENSION: NO. Datapoints not replicated overall.
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)
        
        Recommended bank size is:
            rotation 2 * color 2 = 4, or
            rotation 2 * color 3 = 6, or
            rotation 4 * color 2 = 8.

        Balance the "unseen" level with scaling.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert rotation_bank > 0, "The number of rotation patterns must be greater than 0."
    assert color_bank > 0, "The number of color patterns must be greater than 0."
    assert scaling_high >= scaling_low, "The upper bound of scaling must be greater than the lower bound."
    print("Reverse mode, creating strong unseen level..") if verbose and reverse_test else None

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)
    
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

    pattern_bank = [[angle, color] for angle in angles for color in colors]
    # assign patterns to each client
    client_Count = 0
    print("Showing patterns for each client..\n") if verbose else None

    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):
        print(f"Client: {client_Count}") if verbose else None

        # for training test
        train_pattern = np.random.permutation(pattern_bank).tolist()
        train_pattern = [(float(angle), color) for angle, color in train_pattern]

        scaled_values = np.arange(len(pattern_bank), 0, -1) * np.random.uniform(scaling_low,scaling_high)
        exp_values = np.exp(scaled_values)
        train_prob = exp_values / np.sum(exp_values)

        print("Train bank: ", train_pattern) if verbose else None
        print("Assigned probability: ", train_prob) if verbose else None

        indices = np.arange(len(train_pattern))
        sampled_indices = np.random.choice(indices, size=len(client_data_train['labels']), p=train_prob)
        sampled_pattern = [train_pattern[i] for i in sampled_indices]

        angles_assigned = [item[0] for item in sampled_pattern]
        colors_assigned = [item[1] for item in sampled_pattern]
        client_data_train['features'] = rotate_dataset(client_data_train['features'], angles_assigned)
        client_data_train['features'] = color_dataset(client_data_train['features'], colors_assigned)


        # for testing test
        test_pattern = list(reversed(train_pattern)) if reverse_test else np.random.permutation(pattern_bank).tolist()
        test_pattern = [(float(angle), color) for angle, color in test_pattern]

        scaled_values = np.arange(len(pattern_bank), 0, -1) * np.random.uniform(scaling_low,scaling_high)
        exp_values = np.exp(scaled_values)
        test_prob = exp_values / np.sum(exp_values)

        print("Test bank: ", test_pattern) if verbose else None
        print("Assigned probability: ", test_prob,"\n") if verbose else None

        indices = np.arange(len(test_pattern))
        sampled_indices = np.random.choice(indices, size=len(client_data_test['labels']), p=test_prob)
        sampled_pattern = [test_pattern[i] for i in sampled_indices]

        angles_assigned, colors_assigned = map(list, zip(*sampled_pattern))
        client_data_test['features'] = rotate_dataset(client_data_test['features'], angles_assigned)
        client_data_test['features'] = color_dataset(client_data_test['features'], colors_assigned)

        client_Count += 1

    rearranged_data = []
    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': basic_split_data_train[i]['labels'].detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy()
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
            
    return rearranged_data

def split_trND_teDR_Py(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    scaling_low: float = 0.5,
    scaling_high: float = 0.5,
    reverse_test: bool = False,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A (large in size)
        Testing: B (unseen)
        with distribution difference in P(y)
        for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        scaling_low (float): The lower bound of scaling.
        scaling_high (float): The upper bound of scaling.
        reverse_test (bool): Testing and training use reverse patterns. (used to create strong unseen level)
        verbose (bool): Whether to print the distribution information.

    Description:
        Two probability distributions of the labels will be created for each client, for both training and testing.
        To have a strong unseen level in testing, set reverse_test as True.
        
    Warning:
        EXTENSION: NO. Datapoints not replicated overall.
        SAMPLE: NO. (contrast to SAMPLE)
        CHOICES: YES. Datapoints may repeat for different clients.

        Balance the "unseen" level with scaling.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_high >= scaling_low, "High scaling must be larger than low scaling."
    assert torch.unique(train_labels).size(0) == torch.unique(test_labels).size(0), "Original Dataset Fault."
    label_num = torch.unique(train_labels).size(0)

    print("Reverse mode, creating strong unseen level..") if verbose and reverse_test else None
        
    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number
    rearranged_data = []

    for i in range(client_number):

        print(f"Client: {i}") if verbose else None

        # generate train and test patterns
        train_label_list = torch.randperm(label_num).tolist()
        test_label_list = list(reversed(train_label_list)) if reverse_test else torch.randperm(label_num).tolist()

        # Apply softmax to get the probabilities
        train_scaled_values = np.arange(label_num, 0, -1) * np.random.uniform(scaling_low,scaling_high)
        test_scaled_values = np.arange(label_num, 0, -1) * np.random.uniform(scaling_low,scaling_high)

        train_exp_values = np.exp(train_scaled_values)
        test_exp_values = np.exp(test_scaled_values)

        train_probabilities = train_exp_values / np.sum(train_exp_values)
        test_probabilities = test_exp_values / np.sum(test_exp_values)

        if verbose:
            print("Training dataset label preference: ",train_label_list)
            print("Training dataset probabilities: ",train_probabilities)
            print("Testing dataset label preference: ",test_label_list)
            print("Testing dataset probabilities: ",test_probabilities)

        # Shuffling due to CHOICES
        permuted_train_indices = torch.randperm(len(train_labels))
        permuted_train_features = train_features[permuted_train_indices]
        permuted_train_labels = train_labels[permuted_train_indices]

        permuted_test_indices = torch.randperm(len(test_labels))
        permuted_test_features = test_features[permuted_test_indices]
        permuted_test_labels = test_labels[permuted_test_indices]

        # Initialize the data lists
        client_train_features = []
        client_train_labels = []
        client_test_features = []
        client_test_labels = []

        # Training dataset
        selected_train_points = 0
        train_idx = 0
        while selected_train_points < avg_points_per_client_train:
            current_train_feature = permuted_train_features[train_idx]
            current_train_label = permuted_train_labels[train_idx]

            train_label_index = train_label_list.index(current_train_label.item())
            train_label_prob = train_probabilities[train_label_index]

            # If the random number is less than the probability, we select this point
            if np.random.rand() < train_label_prob:
                client_train_features.append(current_train_feature.unsqueeze(0))
                client_train_labels.append(current_train_label.unsqueeze(0))
                selected_train_points += 1

            train_idx = (train_idx + 1) % len(permuted_train_labels)

        # Testing dataset
        selected_test_points = 0
        test_idx = 0
        while selected_test_points < avg_points_per_client_test:
            current_test_feature = permuted_test_features[test_idx]
            current_test_label = permuted_test_labels[test_idx]

            test_label_index = test_label_list.index(current_test_label.item())
            test_label_prob = test_probabilities[test_label_index]

            # If the random number is less than the probability, we select this point
            if np.random.rand() < test_label_prob:
                client_test_features.append(current_test_feature.unsqueeze(0))
                client_test_labels.append(current_test_label.unsqueeze(0))
                selected_test_points += 1

            test_idx = (test_idx + 1) % len(permuted_test_labels)

        rearranged_data.append({
            'train_features': torch.cat(client_train_features, dim=0).detach().cpu().numpy(),
            'train_labels': torch.cat(client_train_labels, dim=0).detach().cpu().numpy(),
            'test_features': torch.cat(client_test_features, dim=0).detach().cpu().numpy(),
            'test_labels': torch.cat(client_test_labels, dim=0).detach().cpu().numpy(),
        })

    return rearranged_data

def split_trND_teDR_Py_x(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    mixing_num: int = 4,
    scaling_low: float = 0.5,
    scaling_high: float = 0.5,
    verbose: bool = True
) -> list:
    """
    Split the dataset into distributions as:
        Training: A (large in size)
        Testing: B (unseen)
        with distribution difference in P(y|x)
        for A SINGLE CLIENT. (overall skew among clients exists)

    Args:
        train_features (torch.Tensor): The training features.
        train_labels (torch.Tensor): The training labels.
        test_features (torch.Tensor): The testing features.
        test_labels (torch.Tensor): The testing labels.
        client_number (int): The number of clients.
        mixing_num (int): The number of mixed labels.
        scaling_low (float): The lower bound of scaling.
        scaling_high (float): The upper bound of scaling.
        verbose (bool): Whether to print the distribution information.

    Description:
        For each client, train and test datasets are assigned with diff random label-swapping partterns.
        
    Warning:
        EXTENSION: NO. Datapoints not replicated overall.
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

        Balance the "unseen" level with scaling.

        #TODO EXISTING KNOWN BUG for EMNIST (label doesn't start from 0)   

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    
    """
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert 0 <= scaling_low <= scaling_high <= 1, "Scaling factor must be between 0 and 1."
    assert mixing_num > 0, "The number of labels to swap must be larger than 0."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))
    mixing_label_list = np.random.choice(range(0, max_label), mixing_num,replace=False).tolist()

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)
    rearranged_data = []

    print("Showing the label mapping for each client..") if verbose else None


    for i in range(client_number):

        scaling_label = np.random.uniform(scaling_low, scaling_high)
        print("Re-maping possibility for this client is: ", scaling_label) if verbose else None

        # Mapping from original label to the permuted label for training data
        permuted_label_list_train = mixing_label_list.copy()
        np.random.shuffle(permuted_label_list_train)
        label_map_train = {original: permuted for original, permuted in zip(mixing_label_list, permuted_label_list_train)}

        # Mapping from original label to the permuted label for testing data
        permuted_label_list_test = mixing_label_list.copy()
        np.random.shuffle(permuted_label_list_test)
        label_map_test = {original: permuted for original, permuted in zip(mixing_label_list, permuted_label_list_test)}

        print(f'Client {i+1} - Train Label Mapping: {label_map_train}') if verbose else None
        print(f'Client {i+1} - Test Label Mapping: {label_map_test}') if verbose else None

        new_train_labels = basic_split_data_train[i]['labels'].clone()
        new_test_labels = basic_split_data_test[i]['labels'].clone()
        
        for original, permuted_train in label_map_train.items():
            # Replace train labels based on the scaling_label probability
            train_mask = (basic_split_data_train[i]['labels'] == original)
            random_values_train = torch.rand(train_mask.sum().item())
            new_train_labels[train_mask] = torch.where(random_values_train <= scaling_label, permuted_train, original)

        for original, permuted_test in label_map_test.items():
            # Replace test labels based on the scaling_label probability
            test_mask = (basic_split_data_test[i]['labels'] == original)
            random_values_test = torch.rand(test_mask.sum().item())
            new_test_labels[test_mask] = torch.where(random_values_test <= scaling_label, permuted_test, original)

        client_data = {
            'train_features': basic_split_data_train[i]['features'].detach().cpu().numpy(),
            'train_labels': new_train_labels.detach().cpu().numpy(),
            'test_features': basic_split_data_test[i]['features'].detach().cpu().numpy(),
            'test_labels': new_test_labels.detach().cpu().numpy()
        }

        rearranged_data.append(client_data)
    
    return rearranged_data

def split_trND_teDR_Px_y(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    rotation_bank: int = 2,
    color_bank: int = 2,
    rotated_label_number: int = 2,
    colored_label_number: int = 2,
    verbose: bool = True
) -> list:
    '''
    Split the dataset into distributions as:
        Training: A (large in size)
        Testing: B (unseen)
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
        rotated_label_number (int): The number of labels to rotate.
        colored_label_number (int): The number of labels to color.
        verbose (bool): Whether to print the distribution information.

    Description:
        For labels to be rotated and colored, a random choice will be made from the bank.
        And this pattern diffs between train and test.
        
    Warning:
        EXTENSION: NO. Datapoints not replicated overall.
        SAMPLE: YES. Datapoints not ever repeated for different clients.
        CHOICES: NO. (contrast to SAMPLE)

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert rotation_bank > 0, "The number of rotation patterns must be greater than 0."
    assert color_bank > 0, "The number of color patterns must be greater than 0."
    max_label = max(torch.unique(train_labels).size(0), torch.unique(test_labels).size(0))
    assert 0 <= rotated_label_number <= max_label, "Out of range."
    assert 0 <= colored_label_number <= max_label, "Out of range."
    rotated_label_list = np.random.choice(range(0, max_label), rotated_label_number,replace=False).tolist()
    colored_label_list = np.random.choice(range(0, max_label), colored_label_number,replace=False).tolist()
    

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

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
    
    client_Count = 0
    for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

        angle_color_map_train = {i: {'angle': np.random.choice(angles) if i in rotated_label_list else 0.0, 
                            'color': np.random.choice(colors) if i in colored_label_list else 'gray'} 
                    for i in range(max_label)}
        
        print(f'Client {client_Count} train mapping: {angle_color_map_train}') if verbose else None

        angle_color_map_test = {i: {'angle': np.random.choice(angles) if i in rotated_label_list else 0.0, 
                            'color': np.random.choice(colors) if i in colored_label_list else 'gray'} 
                    for i in range(max_label)}
        
        print(f'Client {client_Count} test mapping: {angle_color_map_test}',"\n") if verbose else None

        client_Count += 1

        train_rotations = [angle_color_map_train[label.item()]['angle'] for label in client_data_train['labels']]
        train_colors = [angle_color_map_train[label.item()]['color'] for label in client_data_train['labels']]
        test_rotations = [angle_color_map_test[label.item()]['angle'] for label in client_data_test['labels']]
        test_colors = [angle_color_map_test[label.item()]['color'] for label in client_data_test['labels']]

        client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
        client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)
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
            'test_labels': basic_split_data_test[i]['labels'].detach().cpu().numpy()
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
            
    return rearranged_data