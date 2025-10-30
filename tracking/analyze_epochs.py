import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from SeqTrack.tracking.analysis_results import *  # Import the original analysis functions

def analyze_epochs(base_dir='.', dataset_name='lasot', epochs=range(6, 11)):
    # Prepare data structures to store results
    all_results = {
        'AUC': [],
        'OP50': [],
        'OP75': [],
        'Precision': [],
        'Norm Precision': []
    }
    
    epoch_numbers = []
    
    # Loop through each epoch
    for epoch in epochs:
        epoch_str = f"{epoch:03d}"
        print(f"\n==================== Analyzing Epoch {epoch_str} ====================")
        
        # Create a modified tracker list for this epoch
        trackers = []
        if epoch == 0:
            # Original run (no epoch number)
            trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b256', 
                                      dataset_name=dataset_name, run_ids=None, 
                                      display_name=f'seqtrack_b256_epoch{epoch_str}'))
        else:
            # Epoch-specific runs
            trackers.extend(trackerlist(name='seqtrack', parameter_name=f'seqtrack_b256_{epoch_str:03d}', 
                                      dataset_name=dataset_name, run_ids=None, 
                                      display_name=f'seqtrack_b256_epoch{epoch_str}'))
        
        # Get the dataset
        dataset = get_dataset(dataset_name)
        
        # Print and collect results
        results = print_results(trackers, dataset, dataset_name, merge_results=True, 
                              plot_types=('success', 'prec', 'norm_prec'), 
                              force_evaluation=False)
        
        # Store results for this epoch
        if results:
            epoch_numbers.append(epoch)
            all_results['AUC'].append(results[0][1])  # AUC is the second column in the results
            all_results['OP50'].append(results[1][1])  # OP50 is the second column
            all_results['OP75'].append(results[2][1])  # OP75 is the second column
            all_results['Precision'].append(results[3][1])  # Precision is the second column
            all_results['Norm Precision'].append(results[4][1])  # Norm Precision is the second column
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    metrics = ['AUC', 'OP50', 'OP75', 'Precision', 'Norm Precision']
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i, metric in enumerate(metrics):
        if all_results[metric]:  # Check if we have data for this metric
            plt.plot(epoch_numbers, all_results[metric], 'o-', color=colors[i], label=metric)
    
    plt.title('Tracking Performance Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(epoch_numbers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('analysis_plots', exist_ok=True)
    plot_path = os.path.join('analysis_plots', f'epoch_comparison_{dataset_name}.png')
    plt.savefig(plot_path)
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print(f"{'Epoch':<10} {'AUC':<10} {'OP50':<10} {'OP75':<10} {'Precision':<12} {'Norm Prec':<12}")
    print("-"*60)
    for i, epoch in enumerate(epoch_numbers):
        print(f"{epoch:<10} {all_results['AUC'][i]:<10.4f} {all_results['OP50'][i]:<10.4f} "
              f"{all_results['OP75'][i]:<10.4f} {all_results['Precision'][i]:<12.4f} "
              f"{all_results['Norm Precision'][i]:<12.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Analyze epochs 6 through 10
    analyze_epochs(epochs=range(6, 11))