#!/usr/bin/env python3
"""
Master Evaluation Script for Fetal Brain Harmonization
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("FETAL BRAIN HARMONIZATION - COMPLETE EVALUATION PIPELINE")
print("=" * 80)


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run complete harmonization evaluation pipeline'
    )
    
    parser.add_argument(
        '--test_data',
        default='processed_data_4slice_fixed/test_4slice_data.pkl',
        help='Path to test data pickle file'
    )
    
    parser.add_argument(
        '--weight_dir',
        default='weights/cyclegan_2d',
        help='Directory containing model weights'
    )
    
    parser.add_argument(
        '--output_base',
        default='harmonization_evaluation',
        help='Base output directory for all results'
    )
    
    parser.add_argument(
        '--epochs',
        nargs='+',
        default=['150', '200', 'final'],
        help='Epochs to evaluate'
    )
    
    parser.add_argument(
        '--skip_metrics',
        action='store_true',
        help='Skip quantitative metrics evaluation'
    )
    
    parser.add_argument(
        '--skip_visual',
        action='store_true',
        help='Skip visual comparison generation'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_base) / timestamp
    output_base.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = output_base / 'quantitative_metrics'
    visual_dir = output_base / 'visual_comparison'
    
    print(f"\nOutput directory: {output_base}/")
    print(f"  - Quantitative metrics: {metrics_dir}/")
    print(f"  - Visual comparisons: {visual_dir}/")
    
    # Track success/failure
    results = {}
    
    # Step 1: Quantitative Evaluation
    if not args.skip_metrics:
        cmd_metrics = [
            'python', 'evaluate_fetal_harmonization.py',
            '--test_data', args.test_data,
            '--weight_dir', args.weight_dir,
            '--output_dir', str(metrics_dir),
            '--epochs'
        ] + [str(e) for e in args.epochs if str(e).isdigit()]
        
        results['metrics'] = run_command(
            cmd_metrics,
            "STEP 1: Quantitative Metrics Evaluation"
        )
    else:
        print("\nSkipping quantitative metrics evaluation")
        results['metrics'] = None
    
    # Step 2: Visual Comparison
    if not args.skip_visual:
        cmd_visual = [
            'python', 'create_visual_comparison.py',
            '--test_data', args.test_data,
            '--weight_dir', args.weight_dir,
            '--output_dir', str(visual_dir),
            '--epochs'
        ] + [str(e) for e in args.epochs]
        
        results['visual'] = run_command(
            cmd_visual,
            "STEP 2: Visual Comparison Generation"
        )
    else:
        print("\nSkipping visual comparison generation")
        results['visual'] = None
    
    # Final Summary
    print(f"\n{'='*80}")
    print("EVALUATION PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {output_base}/")
    print(f"\nResults:")
    for step, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        print(f"  {step.upper()}: {status}")
    
    # Check if all succeeded
    if all(r in [True, None] for r in results.values()):
        print(f"\n✓ All evaluation steps completed successfully!")
        print(f"\n View results in: {output_base}/")
        return 0
    else:
        print(f"\n Some evaluation steps failed. Check output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())