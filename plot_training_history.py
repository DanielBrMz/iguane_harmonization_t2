"""
Plot training history from CSV log
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_training_history(csv_path, output_path):
    """
    Plot training losses over epochs
    
    Args:
        csv_path: Path to training_history.csv
        output_path: Where to save figure
    """
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generator loss
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['gen_loss'], linewidth=2, color='blue')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Generator Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Discriminator loss
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['disc_BCH_loss'], linewidth=2, color='red', label='Disc BCH')
    disc_site_cols = [col for col in df.columns if col.startswith('disc_') and col != 'disc_BCH_loss']
    for col in disc_site_cols:
        ax.plot(df['epoch'], df[col], linewidth=1.5, alpha=0.6, label=col.replace('disc_', '').replace('_loss', ''))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Discriminator Losses', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Cycle loss
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['cycle_loss'], linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Cycle Consistency Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Identity loss
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['identity_loss'], linewidth=2, color='purple')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Identity Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to training_history.csv')
    parser.add_argument('--output', required=True, help='Output figure path')
    
    args = parser.parse_args()
    plot_training_history(args.csv, args.output)
