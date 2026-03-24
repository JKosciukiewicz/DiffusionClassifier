import pandas as pd
import numpy as np
import argparse
import os


def clean_csv(
    input_file, output_file=None, fill_method="mean", ignore_prefix="Metadata_"
):
    """
    Cleans a CSV file by:
    1. Removing columns that contain only NaN or inf values
    2. Replacing remaining NaN/inf values with either column mean or a constant value of 0

    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the cleaned CSV file.
                                    If None, will use input_file with '_cleaned' suffix.
        fill_method (str): Method to fill NaN/inf values. Options:
                          'mean': Use column mean
                          'constant': Use constant value 0
        ignore_prefix (str): Prefix for columns to ignore during cleaning (e.g., 'Metadata_')

    Returns:
        pandas.DataFrame: The cleaned DataFrame
    """
    # Set default output file if not provided
    if output_file is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_cleaned{file_ext}"

    # Read the CSV file with low_memory=False to handle mixed data types
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)

    initial_columns = df.columns.tolist()
    initial_column_count = len(initial_columns)

    # Identify metadata columns to ignore during cleaning
    metadata_columns = [col for col in df.columns if col.startswith(ignore_prefix)]
    columns_to_process = [
        col for col in df.columns if not col.startswith(ignore_prefix)
    ]

    if metadata_columns:
        print(
            f"Found {len(metadata_columns)} metadata columns (starting with '{ignore_prefix}') that will be preserved:"
        )
        for col in metadata_columns[:5]:  # Show first 5 metadata columns
            print(f"  - '{col}'")
        if len(metadata_columns) > 5:
            print(f"  - ... and {len(metadata_columns) - 5} more")

    # Identify columns with only NaN values (excluding metadata columns)
    nan_only_columns = [col for col in columns_to_process if df[col].isna().all()]

    # Identify columns with only inf values (excluding metadata columns)
    inf_only_columns = [
        col for col in columns_to_process if (~df[col].isna() & np.isinf(df[col])).all()
    ]

    # Combine both lists to get all columns to drop
    columns_to_drop = list(set(nan_only_columns + inf_only_columns))

    # Drop the identified columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(
            f"Removed {len(columns_to_drop)} columns containing only NaN or inf values:"
        )
        for col in columns_to_drop:
            if col in nan_only_columns:
                print(f"  - '{col}' (NaN only)")
            elif col in inf_only_columns:
                print(f"  - '{col}' (inf only)")
            else:
                print(f"  - '{col}' (both NaN and inf)")
    else:
        print("No columns with only NaN or inf values were found.")

    # Update columns_to_process after dropping columns
    columns_to_process = [col for col in columns_to_process if col in df.columns]

    # Replace remaining NaN and inf values based on chosen method (excluding metadata columns)
    print(
        f"\nReplacing remaining NaN and inf values with {fill_method} (excluding metadata columns)..."
    )

    # Count NaN and inf values before replacement (in non-metadata columns)
    non_metadata_df = df[columns_to_process]
    nan_count_before = non_metadata_df.isna().sum().sum()
    inf_count_before = np.isinf(non_metadata_df).sum().sum()
    total_replacements_needed = nan_count_before + inf_count_before

    if total_replacements_needed > 0:
        # Create a mask for all inf values in non-metadata columns
        for col in columns_to_process:
            # Replace inf with NaN temporarily to handle both cases together
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Apply the chosen fill method
        if fill_method == "mean":
            # Calculate means for each column (ignoring NaN values)
            for col in columns_to_process:
                if df[col].dtype.kind in "if":  # Only process numeric columns
                    col_mean = df[col].mean(skipna=True)
                    if pd.isna(col_mean):
                        # If the mean itself is NaN (possible if all remaining values are NaN)
                        df[col].fillna(0, inplace=True)
                        print(
                            f"  - Column '{col}' has no valid values for mean calculation, using 0 instead"
                        )
                    else:
                        df[col].fillna(col_mean, inplace=True)
                        print(
                            f"  - Column '{col}' NaN/inf values replaced with mean: {col_mean:.4f}"
                        )

        elif fill_method == "constant":
            # Fill all NaN values with 0 (only in non-metadata columns)
            for col in columns_to_process:
                df[col].fillna(0, inplace=True)
            print(
                "  - All NaN/inf values in non-metadata columns replaced with constant value: 0"
            )

        # Count remaining NaN values to verify replacement
        non_metadata_df = df[columns_to_process]
        nan_count_after = non_metadata_df.isna().sum().sum()
        print(f"Replaced values: {total_replacements_needed}")

        if nan_count_after > 0:
            print(
                f"Warning: {nan_count_after} NaN values remain (in non-numeric columns)"
            )
    else:
        print("No NaN or inf values found in the non-metadata columns.")

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"\nCleaned CSV saved to: {output_file}")
    print(f"Original column count: {initial_column_count}")
    print(f"Final column count: {len(df.columns)}")

    return df


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Clean a CSV file by removing columns with only NaN or inf values and replacing remaining NaN/inf values."
    )
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("-o", "--output_file", help="Path to save the cleaned CSV file")
    parser.add_argument(
        "-f",
        "--fill_method",
        choices=["mean", "constant"],
        default="mean",
        help="Method to fill NaN/inf values: 'mean' for column mean or 'constant' for value 0 (default: mean)",
    )

    args = parser.parse_args()

    # Call the function with the provided arguments
    clean_csv(args.input_file, args.output_file, args.fill_method)
