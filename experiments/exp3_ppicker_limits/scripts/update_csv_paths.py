#!/usr/bin/env python3
"""
Script to update file paths in specific CSV columns.
Replaces directories in paths while maintaining original filenames.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Set


def get_filename_from_path(path: str) -> str:
    """Extracts the filename from a complete path."""
    return os.path.basename(path.strip())


def replace_path_directory(old_path: str, new_base_dir: str) -> str:
    """
    Replaces the directory of a path while maintaining the filename.

    Args:
        old_path: Original path
        new_base_dir: New base directory

    Returns:
        New path with updated directory
    """
    if not old_path or not old_path.strip():
        return old_path

    filename = get_filename_from_path(old_path)
    new_path = os.path.join(new_base_dir, filename)
    return new_path


def process_csv(
    input_csv: str,
    output_csv: str,
    columns_to_update: List[str],
    new_directory: str,
    dry_run: bool = False,
    delimiter: str = '\t'
) -> None:
    """
    Processes the CSV updating paths in specified columns.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        columns_to_update: List of column names to update
        new_directory: New base directory for paths
        dry_run: If True, only shows preview without modifying the file
        delimiter: CSV delimiter (default: tab)
    """
    # Validate that the input file exists
    if not os.path.exists(input_csv):
        print(f"Error: File {input_csv} does not exist", file=sys.stderr)
        sys.exit(1)

    # Validate that the new directory exists or create it
    if not dry_run:
        os.makedirs(new_directory, exist_ok=True)

    tab_char = '\t'
    print(f"{'[DRY RUN] ' if dry_run else ''}Processing file: {input_csv}")
    print(f"Columns to update: {', '.join(columns_to_update)}")
    print(f"New directory: {new_directory}")
    print(f"Delimiter: {'TAB' if delimiter == tab_char else repr(delimiter)}")
    print("-" * 80)

    # Counters for statistics
    total_rows = 0
    updated_cells = 0
    preview_count = 0
    max_preview = 5

    # Read and process CSV
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=delimiter)

        # Read header
        header = next(reader)
        print(f"Header found: {len(header)} columns")

        # Identify column indices to update
        column_indices: Set[int] = set()
        for col_name in columns_to_update:
            try:
                col_index = header.index(col_name)
                column_indices.add(col_index)
                print(f"  - '{col_name}' (index {col_index})")
            except ValueError:
                print(
                    f"Warning: Column '{col_name}' not found in CSV", file=sys.stderr)

        if not column_indices:
            print(
                "Error: None of the specified columns exist in the CSV", file=sys.stderr)
            sys.exit(1)

        print("-" * 80)

        # Prepare output file
        if not dry_run:
            outfile = open(output_csv, 'w', encoding='utf-8', newline='')
            writer = csv.writer(outfile, delimiter=delimiter)
            writer.writerow(header)

        # Process rows
        for row in reader:
            total_rows += 1

            # Update specified columns
            updated_row = row.copy()
            for col_idx in column_indices:
                if col_idx < len(row):
                    old_path = row[col_idx]
                    new_path = replace_path_directory(old_path, new_directory)
                    updated_row[col_idx] = new_path

                    if old_path != new_path:
                        updated_cells += 1

                        # Show preview only in dry run
                        if dry_run and preview_count < max_preview:
                            print(f"Column '{header[col_idx]}':")
                            print(f"  Before:  {old_path}")
                            print(f"  After: {new_path}")
                            preview_count += 1

            # Write updated row
            if not dry_run:
                writer.writerow(updated_row)

            # Show progress every 100,000 rows
            if total_rows % 100000 == 0:
                print(f"Processed {total_rows:,} rows...")

        if not dry_run:
            outfile.close()

    # Show summary
    print("-" * 80)
    print(f"{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Updated cells: {updated_cells:,}")

    if dry_run:
        print(f"\n⚠️  DRY RUN MODE: No changes were made to the file")
        print(f"   To apply changes, run the command without --dry-run")
    else:
        print(f"\n✅ Updated file saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Update file paths in specific CSV columns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without modifying (dry run)
  %(prog)s input.csv --columns Density Tomo3D --new-dir /new/path --dry-run
  
  # Update file (replaces the original)
  %(prog)s input.csv --columns Density Micrographs PolyData Tomo3D --new-dir /new/path
  
  # Save to a different file
  %(prog)s input.csv --columns Density --new-dir /new/path --output output.csv
        """
    )

    parser.add_argument(
        'input_csv',
        help='Input CSV file'
    )

    parser.add_argument(
        '-c', '--columns',
        nargs='+',
        required=True,
        help='Column names to update (space separated)'
    )

    parser.add_argument(
        '-d', '--new-dir',
        required=True,
        help='New base directory for paths'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output CSV file (default: replaces input file)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview mode: shows changes without modifying the file'
    )

    parser.add_argument(
        '--delimiter',
        default='\t',
        help='CSV delimiter (default: tab)'
    )

    args = parser.parse_args()

    # If output is not specified, use the same input file
    output_csv = args.output if args.output else args.input_csv

    # If dry run, we don't need an output file
    if args.dry_run:
        output_csv = None

    process_csv(
        input_csv=args.input_csv,
        output_csv=output_csv,
        columns_to_update=args.columns,
        new_directory=args.new_dir,
        dry_run=args.dry_run,
        delimiter=args.delimiter
    )


if __name__ == '__main__':
    main()
