import os
from analysis_results import *  # Import the original analysis functions

def analyze_epochs(base_dir='.', dataset_name='lasot', epochs=range(1, 11)):
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
                                      display_name='seqtrack_b256'))
        else:
            # Epoch-specific runs - use the exact folder name format from your run
            trackers.extend(trackerlist(name='seqtrack', 
                                      parameter_name=f'seqtrack_b256_{epoch:03d}', 
                                      dataset_name=dataset_name, 
                                      run_ids=None, 
                                      display_name=f'seqtrack_b256_epoch{epoch:03d}'))
        
        # Get the dataset
        dataset = get_dataset(dataset_name)
        
        try:
            # Print and collect results
            print(f"\nProcessing tracker: {trackers[0].parameter_name}")
            results = print_results(trackers, dataset, dataset_name, merge_results=True, 
                                  plot_types=('success', 'prec', 'norm_prec'), 
                                  force_evaluation=False)
            
            # Store results for this epoch
            if results and len(results) >= 5:  # Ensure we have all expected metrics
                print(f"Successfully processed epoch {epoch_str}")
                epoch_numbers.append(epoch)
                all_results['AUC'].append(float(results[0][1]))
                all_results['OP50'].append(float(results[1][1]))
                all_results['OP75'].append(float(results[2][1]))
                all_results['Precision'].append(float(results[3][1]))
                all_results['Norm Precision'].append(float(results[4][1]))
            else:
                print(f"Warning: Incomplete results for epoch {epoch_str}")
                
        except Exception as e:
            print(f"Error processing epoch {epoch_str}: {str(e)}")
            continue
    
    if not epoch_numbers:
        print("No valid epochs were processed. Please check your data directories and try again.")
        return
    
    # Print summary table
    print("\n" + "="*80)
    print(f"{'Epoch':<10} {'AUC':<10} {'OP50':<10} {'OP75':<10} {'Precision':<12} {'Norm Prec':<12}")
    print("-"*80)
    for i, epoch in enumerate(epoch_numbers):
        print(f"{epoch:<10} {all_results['AUC'][i]:<10.4f} {all_results['OP50'][i]:<10.4f} "
              f"{all_results['OP75'][i]:<10.4f} {all_results['Precision'][i]:<12.4f} "
              f"{all_results['Norm Precision'][i]:<12.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Analyze epochs 6 through 10
    analyze_epochs(epochs=range(1, 11))