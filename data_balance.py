import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler # Alternative: Undersampling

# --- Configuration ---
FILE_PATH = 'combined.csv'
TARGET_COLUMN = 'target'  # !!! IMPORTANT: Change this to your actual target column name !!!
TEST_SIZE = 0.2  # Proportion of data to use for the test set
RANDOM_STATE = 42  # For reproducibility

# --- Load Data ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded data from '{FILE_PATH}'.")
    print(f"Original DataFrame shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at '{FILE_PATH}'. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Separate Features (X) and Target (y) ---
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the DataFrame.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print(f"\nFeatures shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# --- Check Initial Class Distribution ---
print("\nOriginal class distribution:")
original_counts = Counter(y)
print(original_counts)

# --- Split Data into Training and Testing Sets ---
# IMPORTANT: Balancing should ONLY be applied to the training data
# to avoid data leakage into the test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # Stratify ensures proportions of classes are similar in train/test splits
)

print(f"\nData split into Train/Test:")
print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"  X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

print("\nClass distribution in ORIGINAL training set:")
print(Counter(y_train))
print("\nClass distribution in ORIGINAL test set:")
print(Counter(y_test)) # We don't balance this!

# --- Apply Balancing Technique (SMOTE) to the Training Data ---
print("\nApplying SMOTE (Over-sampling) to the training data...")

# Initialize SMOTE
# You can adjust k_neighbors depending on your data density
smote = SMOTE(random_state=RANDOM_STATE)

# Fit and apply SMOTE only on the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution in RESAMPLED training set:")
resampled_counts = Counter(y_train_resampled)
print(resampled_counts)

print(f"\nShape of resampled training data:")
print(f"  X_train_resampled shape: {X_train_resampled.shape}")
print(f"  y_train_resampled shape: {y_train_resampled.shape}")


# --- Alternative: Undersampling (Example using RandomUnderSampler) ---
# Uncomment the following lines if you want to try undersampling instead of SMOTE
# print("\nApplying RandomUnderSampler (Under-sampling) to the training data...")
# rus = RandomUnderSampler(random_state=RANDOM_STATE)
# X_train_resampled_under, y_train_resampled_under = rus.fit_resample(X_train, y_train)
# print("\nClass distribution in UNDERSAMPLED training set:")
# print(Counter(y_train_resampled_under))
# print(f"\nShape of undersampled training data:")
# print(f"  X_train_resampled_under shape: {X_train_resampled_under.shape}")
# print(f"  y_train_resampled_under shape: {y_train_resampled_under.shape}")
# # If using undersampling, you would use X_train_resampled_under, y_train_resampled_under for training


# --- Ready for ANN Training ---
print("\nData is now balanced and ready for ANN training.")
print("Use X_train_resampled and y_train_resampled for training your ANN.")
print("Use X_test and y_test for evaluating your trained ANN.")

# --- (Optional) Convert resampled data back to DataFrame if needed ---
# df_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
# df_train_resampled[TARGET_COLUMN] = y_train_resampled
# print("\nFirst 5 rows of the resampled training DataFrame:")
# print(df_train_resampled.head())