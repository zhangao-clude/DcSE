import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import random

warnings.filterwarnings("ignore")
species = 'human'
combined_df = pd.read_csv(f"combined_shuffled_{species}.csv")

FEATURE_COL = "sequence"
LABEL_COL = "label"


sequence_counts = combined_df[FEATURE_COL].value_counts()
duplicated_sequences = set(sequence_counts[sequence_counts > 1].index)



saved_splits = 0
base_seed = 42
max_attempts = 1000
attempted_seeds = set()

while saved_splits < 10 and len(attempted_seeds) < max_attempts:
    current_seed = base_seed + len(attempted_seeds)
    attempted_seeds.add(current_seed)

    non_dup_df = combined_df[~combined_df[FEATURE_COL].isin(duplicated_sequences)]

    X_train, X_test, y_train, y_test = train_test_split(
        non_dup_df[FEATURE_COL],
        non_dup_df[LABEL_COL],
        test_size=0.1,
        stratify=non_dup_df[LABEL_COL],
        random_state=current_seed
    )


    train_df = pd.DataFrame({FEATURE_COL: X_train, LABEL_COL: y_train})
    test_df = pd.DataFrame({FEATURE_COL: X_test, LABEL_COL: y_test})


    for dup_seq in duplicated_sequences:

        dup_rows = combined_df[combined_df[FEATURE_COL] == dup_seq]


        random.seed(hash(dup_seq) + current_seed)
        if random.random() < 0.9:  

            dup_train = pd.DataFrame({FEATURE_COL: [dup_seq] * len(dup_rows),
                                      LABEL_COL: dup_rows[LABEL_COL].values})
            train_df = pd.concat([train_df, dup_train], ignore_index=True)
        else:

            dup_test = pd.DataFrame({FEATURE_COL: [dup_seq] * len(dup_rows),
                                     LABEL_COL: dup_rows[LABEL_COL].values})
            test_df = pd.concat([test_df, dup_test], ignore_index=True)


    train_sequences = set(train_df[FEATURE_COL])
    test_sequences = set(test_df[FEATURE_COL])
    overlap = train_sequences.intersection(test_sequences)


    if len(overlap) == 0:
        train_df.to_csv(f"train_{species}_split_{saved_splits}.csv", index=False)
        test_df.to_csv(f"test_{species}_split_{saved_splits}.csv", index=False)
        saved_splits += 1