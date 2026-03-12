import torch
import glob
import os
import numpy as np

def main():
    tasks = ['PVT', 'NBACK', 'FLANKER', 'MATB']
    
    # Ensure Flex-MoE's raw_data folder exists
    save_dir = '/content/Flex-MoE/raw_data/'
    os.makedirs(save_dir, exist_ok=True)

    for task_name in tasks:
        print(f"\n🔄 Stacking dictionaries for {task_name}...")
        
        # FIX: Point exactly to the UGP-01 folder where MAESTRO saved the data!
        data_dir = f'/content/UGP-01/processed_data/{task_name}/**/*.pt'
        all_tensor_files = glob.glob(data_dir, recursive=True)
        print(f"   Found {len(all_tensor_files)} total windows for {task_name}.")

        if len(all_tensor_files) == 0:
            # Let's also check the flat directory just in case there are no subfolders
            data_dir_flat = f'/content/UGP-01/processed_data/{task_name}/*.pt'
            all_tensor_files = glob.glob(data_dir_flat)
            if len(all_tensor_files) == 0:
                print(f"   ⚠️ Skipping {task_name} (No data found).")
                continue
            else:
                print(f"   Found {len(all_tensor_files)} total windows for {task_name} (Flat directory).")

        all_data, all_labels = [], []

        for file_path in all_tensor_files:
            # Extract binary label (0 for Low, 1 for High)
            filename = os.path.basename(file_path)
            try:
                label_str = filename.split('_')[-2] 
                all_labels.append(int(label_str))
            except ValueError:
                continue # Skip files that don't match the naming convention
            
            # Extract data
            obj = torch.load(file_path)
            if isinstance(obj, dict):
                for val in obj.values():
                    if isinstance(val, (torch.Tensor, np.ndarray)):
                        tensor = val
                        break
            else: tensor = obj
                
            if isinstance(tensor, np.ndarray): tensor = torch.tensor(tensor)
            all_data.append(tensor.float())

        if len(all_data) == 0:
            print(f"   ⚠️ Could not extract valid tensors for {task_name}.")
            continue

        # Stack into [Batch, 750, 9]
        master_data = torch.stack(all_data)
        master_labels = torch.tensor(all_labels)

        # Save straight to Flex-MoE's raw_data folder
        torch.save(master_data, os.path.join(save_dir, f'cogbci_data_{task_name}.pt'))
        torch.save(master_labels, os.path.join(save_dir, f'cogbci_labels_{task_name}.pt'))
        print(f"   ✅ Saved {task_name} | Data: {master_data.shape}")

if __name__ == "__main__":
    main()
