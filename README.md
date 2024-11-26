<p align="center">
  <img src="https://github.com/alfredoLimo/ANDA/assets/68495667/593c8ca0-fce0-4fa7-ba73-9d900ce95559" alt="Picture1" width="700"/>
</p>

# :large_blue_circle: ABOUT ANDA
**A** **N**on-IID **D**ata generator supporting **A**ny kind. Generate your non-IID datasets with one line.

# :large_blue_circle: FEATURES
- Repeat your Federated Learning (FL) experiments with non-IID datasets and without saving it!
- Supporting five public datasets: **MNIST**, **EMNIST**, **FMNIST**, **CIFAR10**, and **CIFAR100**
- Supporting six data drifting modes based on three basic types: **static**, **drifting**, **drifting with accumulation**. (details below)
- Supporting five types of non-IID-ness and their mixtures:
  - **feature distribution skew**: P(x)
  - **label distribution skew:** P(y)
  - **concept drift: label condition skew:** P(y|x)
  - **concept drift: feature condition skew:** P(x|y)
  - **quantity skew**
 
# :large_blue_circle: USAGE WITH ONE LINE
- Clone ANDA repo to your working repo.
- Go to the ANDA repo `pip install -r requirements.txt` if needed.
- Create the default static/dynamic non-IID dataset with one line using `load_split_datasets` or `load_split_datasets_dynamic` as following:
```python
from ANDA import anda

# static
new_dataset_static = anda.load_split_datasets()
# dynamic
new_dataset_dynamic = anda.load_split_datasets_dynamic()
```

# :large_blue_circle: STATIC VS DYNAMIC NON-IID
- **Static Non-IID Datasets**: In static non-IID datasets, clients have data distributions that differ from one another. However, within each client, the distribution of the data in the training and testing sets remains consistent and unchanging throughout the training process.
- **Dynamic/Drifting Non-IID Datasets**: In dynamic or drifting non-IID datasets, the data distributions of the training and testing sets can vary, even within the same client. Additionally, the distribution of the training data may shift over time during the training process, leading to potential changes in the model’s learning environment.

<table>
  <tr>
    <th>Training</th>
    <th>Testing</th>
    <th>Module Name</th>
  </tr>
  <tr>
    <td rowspan="2">No drifting (with a relatively large in size dataset)</td>
    <td>Drifting</td>
    <th>trND_teDR</th>
    
  </tr>
  <tr>
    <td>No drifting</td>
    <th>\</th>
    
  </tr>

  
  <tr>
    <td rowspan="2">Drifting with accumulation</td>
    <td>Drifting</td>
    <th>trDA_teDR</th>
    
  </tr>
  <tr>
    <td>No drifting</td>
    <th>trDA_teND</th>
    
  </tr>

  <tr>
    <td rowspan="2">Drifting without accumulation</td>
    <td>Drifting</td>
    <th>trDR_teDR</th>
    
    
  </tr>
  <tr>
    <td>No drifting</td>
    <th>trDR_teND</th>
    
  </tr>
*accumulation: drifting data are <b>appended to</b>, not replacing old datasets.
</table>
 
---

## :large_blue_diamond: MODE static(\\)
> The training set is not drifting (unchanged), as well the testing set.   
The distribution of training is not drifting along epochs, and the dataset is unchanged (and large in size).   
**Training: A (large in size)**   
**Testing: A (same)**
![Picture1](https://github.com/user-attachments/assets/78e40ee6-f16e-4b4b-923e-2b3cf596431c)



| **Non-IID type** | **P(x)**     | **P(y)**           | **P(x\|y)**                            | **P(y\|x)**                          | **Quantity**                      |
|------------------|--------------|--------------------|----------------------------------------|--------------------------------------|-----------------------------------|
| **P(x)**         | `feature_skew` | `feature_label_skew` | /                        | /                        | `feature_skew_unbalanced`           |
| **P(y)**         | /            | `label_skew`         | `feature_condition_skew_with_label_skew` | `label_condition_skew_with_label_skew` | `label_skew_unbalanced`             |
| **P(x\|y)**      | /            | /                  | `feature_condition_skew`                 | /                      | `feature_condition_skew_unbalanced` |
| **P(y\|x)**      | /            | /                  | /                                      | `label_condition_skew`                 | `label_condition_skew_unbalanced`   |
| **Quantity**     | /            | /                  | /                                      | /                                    | `split_unbalanced`                  |
  

### :small_blue_diamond: QUICK START WITH 'AUTO'

`load_split_datasets` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "feature_skew", types of non-IID-ness. [More details](static_mode_guide.md)
- **mode**: str = "auto", using AUTO mode
- **non_iid_level**: str = "medium", auto setting a level for non-IID, "low","medium", or "high"
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results

```python
new_dataset = anda.load_split_datasets(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "feature_skew",
    mode = "auto",
    non_iid_level = "high",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42
)

# output format
# len(new_dataset) = 10 (client_number)
# new_dataset[0].keys() = {
#    'train_features': torch.Tensor
#    'train_labels': torch.Tensor
#    'test_features': torch.Tensor
#    'test_labels': torch.Tensor
#    'cluster': int = -1 (not designed for clustering)
# }
```
Results: (showing data from first four clients, try to repeat it with the same seed)

<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/2cbd40db-0848-4f2f-b564-f686d8e1a4e7" alt="Client 0" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/d1fc756d-8c9f-4e29-9b26-58b315619971" alt="Client 1" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/42edd1ad-4838-4f53-873c-4bb6f5a54b87" alt="Client 2" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/1ae8f2e5-82c9-4868-8493-3fc817b08192" alt="Client 3" width="400"/>

### :small_blue_diamond: CUSTOMIZE WITH 'MANUAL'

`load_split_datasets` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "feature_skew", types of non-IID-ness. [More details](static_mode_guide.md).
- **mode**: str = "manual", using MANUAL mode.
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for chosen non-IID type. [More details](static_mode_guide.md).

```python
new_dataset = anda.load_split_datasets(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "feature_condition_skew",
    mode = "manual",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42,
    set_color = True,
    colors = 3,
    random_mode = True,
    colored_label_number = 4,
)

# output format
# len(new_dataset) = 10 (client_number)
# new_dataset[0].keys() = {
#    'train_features': torch.Tensor
#    'train_labels': torch.Tensor
#    'test_features': torch.Tensor
#    'test_labels': torch.Tensor
#    'cluster': int
# }
```
Results: (showing data from first four clients, try to repeat it with the same seed)

<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/c5aa40c7-e078-4564-b786-8ee3a901e6fa" alt="Client 0" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/708526dc-e7c1-4828-840e-851a1d7e0ab3" alt="Client 1" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/fe9ee7a1-4be5-47bd-b242-bc0a16c96239" alt="Client 2" width="400"/>
<img src="https://github.com/alfredoLimo/ANDA/assets/68495667/a9b75621-5ad5-4d09-be36-b96e38e7def2" alt="Client 3" width="400"/>

### :small_blue_diamond: CLUSTER WITH 'STRICT'

"Strict" functions are provided to create datasets for cluster-oriented tasks, with a rational 'cluster' output for each sub-dataset.

`load_split_datasets` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = ["feature_skew_strict", "label_skew_strict"], ["feature_condition_skew_strict", "label_condition_skew_strict"] types of non-IID-ness. [More details](static_mode_guide.md)
- **mode**: str = "manual", using MANUAL mode.
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for chosen non-IID type. [More details](static_mode_guide.md)

```python
new_dataset = anda.load_split_datasets(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "feature_skew_strict",
    mode = "manual",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42,
    set_rotation: bool = True,
    rotations: int = 2,
    set_color: bool = True,
    colors: int = 2,
)
# Check the kwargs for the other three fns

# output format
# len(new_dataset) = 10 (client_number)
# new_dataset[0].keys() = {
#    'train_features': torch.Tensor
#    'train_labels': torch.Tensor
#    'test_features': torch.Tensor
#    'test_labels': torch.Tensor
#    'cluster': int (cluster identity)
# }
```
---

## :large_blue_diamond: MODE trND_teDR
> The training set is not drifting (unchanged), but the testing set drifted.   
The distribution of training is not drifting along epochs, and the dataset is unchanged (and large in size).   
The distribution of testing is drifting.   
**Training: A (large in size)**   
**Testing: B (unseen)**  
![Picture2](https://github.com/user-attachments/assets/0c866768-e135-4d40-bd79-4ba29d079166)


`load_split_datasets_dynamic` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "Px", types of non-IID-ness. ["Px","Py","Px_y","Py_x"]
- **drfting_type**: str = "trND_teDR", trND_teDR mode
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for trND_teDR. [More details](dynamic_mode_guide.md).
```python
new_dataset = anda.load_split_datasets_dynamic(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "Px",
    drfting_type = "trND_teDR",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42
)

# output format
# len(new_dataset) = 10 (client_number)
# new_dataset[0].keys() = {
#    'train_features': torch.Tensor
#    'train_labels': torch.Tensor
#    'test_features': torch.Tensor
#    'test_labels': torch.Tensor
# }
```
Results: (showing data from the first client, both training and testing sets)

<img src="https://github.com/user-attachments/assets/40155e5d-d05a-4243-8cb5-d01c642bf0fa" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/00f66663-757e-4ab5-acd9-c0a462f9e1b2" alt="Client 0" width="400"/>


---

## :large_blue_diamond: MODE trDA_teDR
> The training set is drifting with accumulation, and the testing set drifted.   
The distribution of training is drifting along epochs and accumulating.   
The distribution of testing drifted (unseen to the client).   
**Training: A-A-AB-ABB-ABB-ABBB-ABBBC-ABBBC**   
**Testing: D (unseen)**
![Picture3](https://github.com/user-attachments/assets/87feddd7-5164-4e0d-bd8f-7b4b72b4302a)

`load_split_datasets_dynamic` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "Px", types of non-IID-ness. ["Px","Py","Px_y","Py_x"]
- **drfting_type**: str = "trDA_teDR", trDA_teDR mode
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for trDA_teDR. [More details](dynamic_mode_guide.md).
```python
new_dataset = anda.load_split_datasets_dynamic(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "Py",
    drfting_type = "trDA_teDR",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42
)

# output format
# new_dataset[0].keys() = {
#    'train': bool, True for training set, False for testing set
#    'features': torch.Tensor
#    'labels': torch.Tensor
#    'client_number': int, client number
#    'epoch_locker_indicator': float [0.0, 1.0], indicating data updating time
#                0.0: the first epoch, 1.0: the last epoch (percentage)
#    'epoch_locker_order': int, the order of the epoch_locker_indicator
#    'cluster': int, the cluster identity of the sub-dataset
# }
```
Results: (showing data from the first client, both training (rounds 1,2,3) and testing sets)

<img src="https://github.com/user-attachments/assets/b3c4a02a-ea6d-45c1-9477-66704f6252aa" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/fdba99de-83cf-4cd6-8e84-a018e0c2d71f" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/831601e8-d86f-41cd-9582-63f630612387" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/52c5ca84-3b0e-4b29-9c32-52b1e9df5ee3" alt="Client 0" width="400"/>

---

## :large_blue_diamond: MODE trDA_teND
> The training set is drifting with accumulation, and the testing set is not drifting.   
The distribution of training is drifting along epochs and accumulating.   
The distribution of testing is not drifting (seen at least once).   
**Training: A-A-AB-ABB-ABB-ABBB-ABBBC-ABBBC …**   
**Testing: A/B/C (seen at least once)**   
![Picture4](https://github.com/user-attachments/assets/d2faef95-0c55-44e9-b508-a386d2ca1b6e)

`load_split_datasets_dynamic` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "Px", types of non-IID-ness. ["Px","Py","Px_y","Py_x"]
- **drfting_type**: str = "trDA_teND", trDA_teND mode
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for trDA_teND. [More details](dynamic_mode_guide.md).
```python
new_dataset = anda.load_split_datasets_dynamic(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "Px_y",
    drfting_type = "trDA_teND",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42
)

# output format
# new_dataset[0].keys() = {
#    'train': bool, True for training set, False for testing set
#    'features': torch.Tensor
#    'labels': torch.Tensor
#    'client_number': int, client number
#    'epoch_locker_indicator': float [0.0, 1.0], indicating data updating time
#                0.0: the first epoch, 1.0: the last epoch (percentage)
#    'epoch_locker_order': int, the order of the epoch_locker_indicator
#    'cluster': int, the cluster identity of the sub-dataset
# }
```
Results: (showing data from the first client, both training (rounds 1,2,3) and testing sets)

<img src="https://github.com/user-attachments/assets/afba0b47-05cf-4b84-8371-958dfabceb3a" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/8eb89e7e-744f-4766-8fe0-3df8d280d0fe" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/f44170b2-65a4-4cdc-84f7-a190eaa5d3ff" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/750246a8-e326-472a-aebf-44b62ea10da6" alt="Client 0" width="400"/>

---

## :large_blue_diamond: MODE trDR_teDR
> The training set is drifting without accumulation, and the testing set drifted.   
The distribution of training is drifting along epochs.
The distribution of testing drifted (unseen to the client).   
**Training: A-A-B-B-B-C-C-C-A-A-D-D**   
**Testing: E (unseen)**   
![Picture5](https://github.com/user-attachments/assets/65438c66-6cfc-46a1-95a8-9ad24f102a6f)

`load_split_datasets_dynamic` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "Px", types of non-IID-ness. ["Px","Py","Px_y","Py_x"]
- **drfting_type**: str = "trDR_teDR", trDR_teDR mode
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for trDR_teDR. [More details](dynamic_mode_guide.md).
```python
new_dataset = anda.load_split_datasets_dynamic(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "Py_x",
    drfting_type = "trDR_teDR",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42
)

# output format
# new_dataset[0].keys() = {
#    'train': bool, True for training set, False for testing set
#    'features': torch.Tensor
#    'labels': torch.Tensor
#    'client_number': int, client number
#    'epoch_locker_indicator': float [0.0, 1.0], indicating data updating time
#                0.0: the first epoch, 1.0: the last epoch (percentage)
#    'epoch_locker_order': int, the order of the epoch_locker_indicator 
#    'cluster': int, the cluster identity of the sub-dataset
# }
```
Results: (showing data from the first client, both training (rounds 1,2,3) and testing sets)

<img src="https://github.com/user-attachments/assets/039331c9-2af0-4e34-85b3-b85410aff94f" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/489d9db4-261a-4442-8bdb-a4107da96e72" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/a94b362a-5da0-486b-b1c9-60acac08604f" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/c8f23411-1400-48ba-a29f-dae2e9acdb4a" alt="Client 0" width="400"/>

---

## :large_blue_diamond: MODE trDR_teND
> The training set is drifting without accumulation, and the testing set is not drifting.   
The distribution of training is drifting along epochs.   
The distribution of testing is not drifting (seen at least once).   
**Training: A-A-B-B-B-C-C-C-A-A-D-D …**   
**Testing: A/B/C/D (seen at least once)**  
![Picture6](https://github.com/user-attachments/assets/d11f2f12-9ab2-405e-af88-14a0d34cdc3a)

`load_split_datasets_dynamic` **parameters**

- **dataset_name**: str = "MNIST", "EMNIST", "FMNIST", "CIFAR10", or "CIFAR100"
- **client_number**: int = 10, number of clients/sub-datasets
- **non_iid_type**: str = "Px", types of non-IID-ness. ["Px","Py","Px_y","Py_x"]
- **drfting_type**: str = "trDR_teND", trDR_teND mode
- **verbose**: bool = False, show generated feature details if any
- **count_labels**: bool = False, count label details for each client
- **plot_clients**: bool = False, plot clients' data distribution
- **random_seed**: int = 42, a random seed to repeat your results
- **\*\*kwargs**: customized parameters for trDR_teND. [More details](dynamic_mode_guide.md).
```python
new_dataset = anda.load_split_datasets_dynamic(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "Px",
    drfting_type = "trDR_teND",
    verbose = True,
    count_labels = True,
    plot_clients = True,
    random_seed = 42
)

# output format
# new_dataset[0].keys() = {
#    'train': bool, True for training set, False for testing set
#    'features': torch.Tensor
#    'labels': torch.Tensor
#    'client_number': int, client number
#    'epoch_locker_indicator': float [0.0, 1.0], indicating data updating time
#                0.0: the first epoch, 1.0: the last epoch (percentage)
#    'epoch_locker_order': int, the order of the epoch_locker_indicator 
#    'cluster': int, the cluster identity of the sub-dataset
# }
```
Results: (showing data from the first client, both training (rounds 1,2,3) and testing sets)

<img src="https://github.com/user-attachments/assets/10093549-43c8-4298-8378-20afd293f539" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/cf2a4731-1819-4379-bc2a-84662504ec0c" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/f8654d18-b607-4705-9727-8bf99c17178e" alt="Client 0" width="400"/>
<img src="https://github.com/user-attachments/assets/7d218f7b-4ae3-4401-ab68-71400a3975e5" alt="Client 0" width="400"/>

# :large_blue_circle: MORE ON NON-IID
[Independent and identically distributed (IID) random variables](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)

[Federated learning on non-IID data: A survey](https://www.sciencedirect.com/science/article/pii/S0925231221013254)

# :large_blue_circle: TODO LIST
- Scripts for quick start with FL libs (PySyft, Flower, etc.)
