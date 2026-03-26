#!/usr/bin/env python3
"""
Filter recommendation JSON files to keep only successful queries common to both models.

This script:
1. Loads Gemini and GPT-4o recommendation files
2. Filters for success=True with valid rec_cities
3. Finds query_ids that are successful in BOTH files
4. Keeps only those common successful queries in each file
5. Saves the filtered results back to the original files
"""

import json
from pathlib import Path


def filter_common_successful_queries(gemini_file: Path, gpt_file: Path):
    """
    Filter both JSON files to keep only successful queries common to both.

    Args:
        gemini_file: Path to Gemini recommendations JSON
        gpt_file: Path to GPT-4o recommendations JSON
    """
    print("="*80)
    print("Filtering Recommendation Files for Common Successful Queries")
    print("="*80)

    # Load both files
    print(f"\nLoading files...")
    print(f"  Gemini: {gemini_file}")
    print(f"  GPT-4o: {gpt_file}")

    with open(gemini_file, 'r', encoding='utf-8') as f:
        gemini_data = json.load(f)

    with open(gpt_file, 'r', encoding='utf-8') as f:
        gpt_data = json.load(f)

    print(f"\n✓ Loaded successfully")
    print(f"  Gemini original count: {len(gemini_data)}")
    print(f"  GPT-4o original count: {len(gpt_data)}")

    # Filter for success=True with valid rec_cities
    gemini_success = [
        r for r in gemini_data
        if r.get('success', False) and 'rec_cities' in r and r['rec_cities']
    ]
    gpt_success = [
        r for r in gpt_data
        if r.get('success', False) and 'rec_cities' in r and r['rec_cities']
    ]

    print(f"\n✓ Filtered for successful queries")
    print(f"  Gemini successful: {len(gemini_success)}")
    print(f"  GPT-4o successful: {len(gpt_success)}")

    # Get query_ids that are successful in both
    gemini_query_ids = {r['query_id'] for r in gemini_success}
    gpt_query_ids = {r['query_id'] for r in gpt_success}

    common_query_ids = gemini_query_ids & gpt_query_ids

    print(f"\n✓ Found common successful query_ids: {len(common_query_ids)}")

    if not common_query_ids:
        print("\n⚠ Warning: No common successful queries found!")
        return

    # Filter to keep only common queries
    gemini_filtered = [r for r in gemini_success if r['query_id'] in common_query_ids]
    gpt_filtered = [r for r in gpt_success if r['query_id'] in common_query_ids]

    # Sort by query_id for consistency
    gemini_filtered.sort(key=lambda x: str(x['query_id']))
    gpt_filtered.sort(key=lambda x: str(x['query_id']))

    # Create backup of original files
    backup_dir = gemini_file.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    gemini_backup = backup_dir / f"{gemini_file.stem}_backup.json"
    gpt_backup = backup_dir / f"{gpt_file.stem}_backup.json"

    print(f"\n✓ Creating backups...")
    print(f"  {gemini_backup}")
    print(f"  {gpt_backup}")

    with open(gemini_backup, 'w', encoding='utf-8') as f:
        json.dump(gemini_data, f, indent=2, ensure_ascii=False)

    with open(gpt_backup, 'w', encoding='utf-8') as f:
        json.dump(gpt_data, f, indent=2, ensure_ascii=False)

    # Save filtered results
    print(f"\n✓ Saving filtered results...")
    with open(gemini_file, 'w', encoding='utf-8') as f:
        json.dump(gemini_filtered, f, indent=2, ensure_ascii=False)

    with open(gpt_file, 'w', encoding='utf-8') as f:
        json.dump(gpt_filtered, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("Summary")
    print("="*80)
    print(f"Original Gemini queries: {len(gemini_data)}")
    print(f"Original GPT-4o queries: {len(gpt_data)}")
    print(f"Successful Gemini queries: {len(gemini_success)}")
    print(f"Successful GPT-4o queries: {len(gpt_success)}")
    print(f"Common successful queries: {len(common_query_ids)}")
    print(f"\n✓ Both files now contain {len(common_query_ids)} common successful queries")
    print(f"✓ Original files backed up to: {backup_dir}")

    # Show sample of common query IDs
    sample_ids = sorted(list(common_query_ids))[:10]
    print(f"\nSample query_ids: {sample_ids}")
    if len(common_query_ids) > 10:
        print(f"... and {len(common_query_ids) - 10} more")


if __name__ == "__main__":
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent / "data" / "conv-trs" / "ecir-2026" / "rec-llm"
    gemini_file = base_dir / "gemini_2_5_flash_recommendations.json"
    gpt_file = base_dir / "gpt_4o_recommendations.json"

    # Check if files exist
    if not gemini_file.exists():
        print(f"✗ Error: Gemini file not found: {gemini_file}")
        exit(1)

    if not gpt_file.exists():
        print(f"✗ Error: GPT-4o file not found: {gpt_file}")
        exit(1)

    # Run the filtering
    filter_common_successful_queries(gemini_file, gpt_file)

