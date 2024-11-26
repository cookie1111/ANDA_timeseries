# Dynamic mode supported types of non-IID-ness

## Mode **`trND_teDR`**

Resembles static (\\) mode as the training set is not changing during all epochs. More details about the color/rotation can be found in the [static mode guide](static_mode_guide.md).

### non_iid_type = `Px`
> Creating feature-skewed (Px) sub-datasets with (image) rotation and coloring.
> A pattern bank will be created based on #rotation and #color. Each DATAPOINT applies one of the patterns in the bank. Each sub-dataset choose one and only distribution of the patterns.
- `rotation_bank: int` The number of rotation patterns. **1** as no rotation.
- `color_bank: int` The number of color patterns. **1** as no color.
- `scaling_low, scaling_high: float` A scaling factor range (randomly uniformly chosen between) for non-IID-ness with a softmax function.
- `reverse_test: bool` Reverse the patterns from the training set for the testing set. (creating strong unseen level)
  
### non_iid_type = `Py`
> Creating label-skewed (Py) sub-datasets.
> Two probability distributions of the labels will be created for each client, for both training and testing.
- `scaling_low, scaling_high: float` A scaling factor range (randomly uniformly chosen between) for non-IID-ness with a softmax function.
- `reverse_test: bool` Reverse the patterns from the training set for the testing set. (creating strong unseen level)

### non_iid_type = `Py_x`
> Creating feature condition skewed (P(y|x)) sub-datasets.
> For each client, train and test datasets are assigned with diff random label-swapping partterns.
- `mixing_num: int` The number of mixed labels. A list of classes (#len = mixing_num) will be generated as the only swapping pool.
- `scaling_low, scaling_high: float` A scaling factor range (randomly uniformly chosen between) for non-IID-ness with a softmax function.

### non_iid_type = `Px_y`
> Creating feature condition skewed (P(x|y)) sub-datasets.
> Randomly chosen #rotated_label_number and #colored_label_number of classes will be rotated/colored. And the pattern is randomly generated based on #rotation_bank and #color_bank.
- `rotation_bank: int` The number of rotation patterns. **1** as no rotation.
- `color_bank: int` The number of color patterns. **1** as no color.
- `rotated_label_number: int` The number of labels (classes) to rotate.
- `colored_label_number: int` The number of labels (classes) to color.

## Mode **`trDA_teDR`** **`trDA_teND`** **`trDR_teDR`** **`trDR_teND`**

All the four modes share the same following parameters (**DA**) for the dynamic setting. (non-IID type agnostic)

#### `DA_dataset_scaling: float`
The scaling factor for the training dataset. (size = original size * dataset_scaling)
Different non-IID types take various ways to drift datasets. Fine-tune for an ideal size.
#### `DA_epoch_locker_num: int`
The number of dataset growth/changing stages during overall training.

E.g., 2 means the dataset grows/changes once (two stages), and an indicator float=0, 0.5 will be labeled to the sub-dataset to let known when to grow/change.

(Accordingly, 3 gives float=0, 0.33, 0.67)

Use `DA_epoch_locker_num` in your training to apply the correct datasets w.r.t. the locker tag.
#### `DA_random_locker: bool`
If `True`, the locker float is randomly generated. Otherwise, it is uniformly generated.
#### `DA_max_dist: int`
The maximum distribution types during overall training.

Setting to 1 means the dataset distribution will not change during the training.

Setting to 3 means the dataset distribution will change **at maximum** three times during the training.

**Note**: Upper limit standing for certain types of non-IID setting.
#### `DA_continual_divergence: bool`
Whether the distribution drifts continually.

E.g. when `True`, (example with accumulation)

drifting as [A]-[AB]-[ABC]-[ABCD]-[ABCDE] is VALID. (continual)

drifting as [A]-[AB]-[ABC]-[ABCD]-[ABCDA] is INVALID. (back to dist A)

---

### non_iid_type = `Px`
> Creating feature-skewed (Px) sub-datasets with (image) rotation and coloring.
> A pattern bank will be created based on #rotation and #color.
> Each pattern is considered as a type of P(x) distribution (A/B/C ...).
> Each subset taking a certain pattern for all its datapoints.
- `rotation_bank: int` The number of rotation patterns. **1** as no rotation.
- `color_bank: int` The number of color patterns. **1** as no color.

**Note**: The overall training set is chunked for sub stages (Last stage grows to full size). Tune `DA_dataset_scaling` for an ideal size.

### non_iid_type = `Py`
> One subset has #classes_per_set classes as a type of P(y) distribution (A/B/C ...).
- `py_bank: int` The number of different Py distributions. Each distribution has #classes_per_set classes.
- `classes_per_set: int` The number of classes in one Py distribution.

**Note**: scaling not supported.

### non_iid_type = `Py_x`
> #mixing_num classes will be randomly selected for label-swapping.
>One permutation of classes as a type of P(y|x) distribution (A/B/C ...).
>E.g. mixing_num=3, classes [0,5,7] are selected. The swap_bank will be:
>[0,5,7],[0,7,5],[5,0,7],[5,7,0],[7,0,5],[7,5,0]
- `mixing_num: int` The number of mixed classes. (Classes randomly chosen)

**Note**: The overall training set is chunked for sub stages (Last stage grows to full size). Tune `DA_dataset_scaling` for an ideal size.

### non_iid_type = `Px_y`
> A Px pattern bank will be created based on #rotation and #color.
> A Pyx pattern bank (len = #pyx_pattern_bank_num) will be created as a type of P(x|y) distribution (A/B/C ...).
> Each pattern chooses a Px pattern and applies to #targeted_class_number classes (randomly selected).
- `rotation_bank: int` The number of rotation patterns. **1** as no rotation.
- `color_bank: int` The number of color patterns. **1** as no color.
- `pyx_pattern_bank_num: int` The number of different Pyx distributions. Each distribution has #targeted_class_number classes.
- `targeted_class_number: int` The number of classes in one Pyx distribution.

**Note**: The overall training set is chunked for sub stages (Last stage grows to full size). Tune `DA_dataset_scaling` for an ideal size.