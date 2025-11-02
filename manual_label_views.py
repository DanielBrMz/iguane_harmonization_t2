"""
Interactive manual labeling interface for view classification
Simple, fast keyboard-based labeling
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys


class ViewLabeler:
    def __init__(self, data_path, output_path, start_index=0):
        """
        Interactive labeling interface
        
        Controls:
        - Press 's' for SAGITTAL (side view)
        - Press 'a' for AXIAL/CORONAL (top-down/front view)
        - Press 'u' for UNKNOWN/UNCLEAR
        - Press 'b' to go BACK one image
        - Press 'q' to QUIT and save progress
        """
        print("Loading data...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.images = self.data['images']
        self.sites = self.data['site']
        self.n_images = len(self.images)
        self.output_path = output_path
        
        # Initialize or load existing labels
        if 'view' in self.data and start_index > 0:
            self.labels = self.data['view'].tolist()
            print(f"Loaded existing labels, resuming from index {start_index}")
        else:
            self.labels = ['unlabeled'] * self.n_images
        
        self.current_index = start_index
        
        # Statistics
        self.label_counts = {'sagittal': 0, 'axial': 0, 'unknown': 0, 'unlabeled': self.n_images}
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n" + "="*70)
        print("MANUAL VIEW LABELING")
        print("="*70)
        print("\nKEYBOARD CONTROLS:")
        print("  's' = SAGITTAL (side view, profile)")
        print("  'a' = AXIAL/CORONAL (top-down or front view)")
        print("  'u' = UNKNOWN (unclear or poor quality)")
        print("  'b' = BACK (go to previous image)")
        print("  'q' = QUIT and save progress")
        print("\nView identification guide:")
        print("  SAGITTAL: Side profile, one hemisphere visible, C-shaped")
        print("  AXIAL/CORONAL: Both hemispheres visible, circular or butterfly-shaped")
        print("="*70)
        print(f"\nStarting at image {start_index}/{self.n_images}")
        print("Close the window or press 'q' to save and quit\n")
        
        self.show_image()
        plt.show()
    
    def show_image(self):
        """Display current image with info"""
        self.ax.clear()
        
        if self.current_index >= self.n_images:
            self.ax.text(0.5, 0.5, 'LABELING COMPLETE!\n\nPress Q to save and quit',
                        ha='center', va='center', fontsize=20, fontweight='bold')
            self.ax.axis('off')
            self.fig.canvas.draw()
            return
        
        # Get current image
        img = self.images[self.current_index, :, :, 0]
        site = self.sites[self.current_index]
        current_label = self.labels[self.current_index]
        
        # Display image
        self.ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        
        # Add info text
        progress = f"Image {self.current_index + 1}/{self.n_images} ({100*(self.current_index+1)/self.n_images:.1f}%)"
        site_info = f"Site: {site}"
        label_info = f"Current label: {current_label}"
        
        # Color code by current label
        label_colors = {
            'sagittal': 'green',
            'axial': 'blue',
            'unknown': 'orange',
            'unlabeled': 'red'
        }
        color = label_colors.get(current_label, 'white')
        
        title = f"{progress}\n{site_info}\n{label_info}"
        self.ax.set_title(title, fontsize=14, fontweight='bold', 
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        # Show statistics
        stats_text = (f"Labeled: {self.current_index - self.label_counts['unlabeled']}/{self.n_images}\n"
                     f"Sagittal: {self.label_counts['sagittal']}\n"
                     f"Axial: {self.label_counts['axial']}\n"
                     f"Unknown: {self.label_counts['unknown']}")
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Controls reminder
        controls = "S=Sagittal | A=Axial/Coronal | U=Unknown | B=Back | Q=Quit"
        self.ax.text(0.5, 0.02, controls, transform=self.ax.transAxes,
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        self.ax.axis('off')
        self.fig.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard input"""
        key = event.key.lower()
        
        if key == 'q':
            self.save_and_quit()
            return
        
        if key == 'b':
            # Go back
            if self.current_index > 0:
                # Undo previous label
                old_label = self.labels[self.current_index - 1]
                if old_label != 'unlabeled':
                    self.label_counts[old_label] -= 1
                    self.label_counts['unlabeled'] += 1
                
                self.labels[self.current_index - 1] = 'unlabeled'
                self.current_index -= 1
                self.show_image()
            return
        
        if key in ['s', 'a', 'u']:
            # Update label
            label_map = {'s': 'sagittal', 'a': 'axial', 'u': 'unknown'}
            new_label = label_map[key]
            
            old_label = self.labels[self.current_index]
            self.labels[self.current_index] = new_label
            
            # Update counts
            if old_label != 'unlabeled':
                self.label_counts[old_label] -= 1
            else:
                self.label_counts['unlabeled'] -= 1
            
            self.label_counts[new_label] += 1
            
            # Move to next
            self.current_index += 1
            
            # Auto-save every 50 images
            if self.current_index % 50 == 0:
                self.save_progress()
            
            self.show_image()
    
    def save_progress(self):
        """Save current progress"""
        self.data['view'] = np.array(self.labels, dtype=object)
        
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.data, f)
        
        print(f"Progress saved: {self.current_index}/{self.n_images} images labeled")
    
    def save_and_quit(self):
        """Save and close"""
        self.save_progress()
        
        print("\n" + "="*70)
        print("LABELING SESSION COMPLETE")
        print("="*70)
        print(f"\nTotal images labeled: {self.current_index}/{self.n_images}")
        print(f"Sagittal: {self.label_counts['sagittal']}")
        print(f"Axial/Coronal: {self.label_counts['axial']}")
        print(f"Unknown: {self.label_counts['unknown']}")
        print(f"Remaining: {self.label_counts['unlabeled']}")
        
        if self.label_counts['unlabeled'] > 0:
            print(f"\nTo resume labeling, run:")
            print(f"python manual_label_views.py {sys.argv[1]} --start {self.current_index}")
        else:
            print("\n✓ All images labeled!")
            print(f"\nSaved to: {self.output_path}")
            print("\nNext step: Split dataset by view")
            print(f"python split_by_view.py {self.output_path}")
        
        print("="*70)
        
        plt.close(self.fig)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual view labeling interface')
    parser.add_argument('data_path', help='Path to training data pickle')
    parser.add_argument('--output', default=None, 
                       help='Output path (default: input_path with _labeled suffix)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index for resuming labeling')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.data_path.replace('.pkl', '_labeled.pkl')
    
    labeler = ViewLabeler(args.data_path, args.output, args.start)