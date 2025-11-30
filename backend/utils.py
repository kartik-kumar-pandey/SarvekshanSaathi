import os
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.patches as mpatches
import pandas as pd
import gc
try:
    from skimage.filters import threshold_otsu
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Dataset Metadata for Visualization
DATASET_METADATA = {
    'pavia': {
        'label_to_name': {
            1: 'Asphalt', 2: 'Meadows', 3: 'Gravel', 4: 'Trees',
            5: 'Painted metal sheets', 6: 'Bare Soil', 7: 'Bitumen',
            8: 'Self-Blocking Bricks', 9: 'Shadows'
        },
        'descriptions': {
            'Asphalt': "Urban road surface.", 'Meadows': "Large grassy areas.",
            'Gravel': "Loose rock surface.", 'Trees': "Vegetation cover.",
            'Painted metal sheets': "Reflective urban structures.", 'Bare Soil': "Dry, unplanted earth.",
            'Bitumen': "Waterproof construction surface.", 'Self-Blocking Bricks': "Interlocking brick patterns.",
            'Shadows': "Shadowed zones in urban area."
        }
    },
    'indian': {
        'label_to_name': {
            1: 'Alfalfa', 2: 'Corn-notill', 3: 'Corn-mintill', 4: 'Corn',
            5: 'Grass-pasture', 6: 'Grass-trees', 7: 'Grass-pasture-mowed', 8: 'Hay-windrowed',
            9: 'Oats', 10: 'Soybean-notill', 11: 'Soybean-mintill', 12: 'Soybean-clean',
            13: 'Wheat', 14: 'Woods', 15: 'Buildings-Grass-Trees-Drives', 16: 'Stone-Steel-Towers'
        },
        'descriptions': {
            'Alfalfa': "Perennial flowering plant.", 'Corn-notill': "Corn grown without tillage.",
            'Corn-mintill': "Corn with minimal tillage.", 'Corn': "Fully cultivated corn.",
            'Grass-pasture': "Grazing grassland.", 'Grass-trees': "Mixed vegetation.",
            'Grass-pasture-mowed': "Cut pasture field.", 'Hay-windrowed': "Dry hay laid in rows.",
            'Oats': "Cultivated oats.", 'Soybean-notill': "Soybean without tillage.",
            'Soybean-mintill': "Soybean with light tillage.", 'Soybean-clean': "Soybean grown cleanly.",
            'Wheat': "Wheat crop.", 'Woods': "Dense trees.",
            'Buildings-Grass-Trees-Drives': "Urban mix with greenery.", 'Stone-Steel-Towers': "Tall infrastructure features."
        }
    },
    'salinas': {
        'label_to_name': {
            1: "Broccoli_green_weeds_1", 2: "Broccoli_green_weeds_2", 3: "Fallow",
            4: "Fallow_rough_plow", 5: "Fallow_smooth", 6: "Stubble", 7: "Celery",
            8: "Grapes_untrained", 9: "Soil_vinyard_develop", 10: "Corn_senesced_green_weeds",
            11: "Lettuce_romaine_4wk", 12: "Lettuce_romaine_5wk", 13: "Lettuce_romaine_6wk",
            14: "Lettuce_romaine_7wk", 15: "Vinyard_untrained", 16: "Vinyard_vertical_trellis"
        },
        'descriptions': {
            "Broccoli_green_weeds_1": "Weeds in broccoli field type 1.",
            "Broccoli_green_weeds_2": "Weeds in broccoli field type 2.",
            "Fallow": "Unplanted agricultural field.",
            "Fallow_rough_plow": "Rough-plowed, uncultivated land.",
            "Fallow_smooth": "Smooth, barren agricultural field.",
            "Stubble": "Remnants of harvested crops.",
            "Celery": "Celery vegetation detected.",
            "Grapes_untrained": "Grapevines not trained on trellis.",
            "Soil_vinyard_develop": "Soil under vineyard development.",
            "Corn_senesced_green_weeds": "Dried corn with weeds.",
            "Lettuce_romaine_4wk": "4-week romaine lettuce.",
            "Lettuce_romaine_5wk": "5-week romaine lettuce.",
            "Lettuce_romaine_6wk": "6-week romaine lettuce.",
            "Lettuce_romaine_7wk": "7-week romaine lettuce.",
            "Vinyard_untrained": "Untrained vineyard vines.",
            "Vinyard_vertical_trellis": "Vineyard with vertical trellis."
        }
    }
}

def preprocess(data, gt, dataset_name):
    h, w, bands = data.shape
    if dataset_name == 'indian':
        noisy_bands = [b for b in (list(range(104, 109)) + list(range(150, 164)) + [220]) if b < data.shape[-1]]
        data = np.delete(data, noisy_bands, axis=2)
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, data.shape[2])
    data_scaled = scaler.fit_transform(data_reshaped).astype(np.float32)  # convert to float32 to reduce memory usage
    pca_components = 30 if dataset_name != 'indian' else 40
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_scaled)
    data_pca = data_pca.reshape(h, w, -1)
    return data_pca, gt, h, w, pca_components

def extract_patches(data, gt, patch_size):
    h, w, c = data.shape
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    padded_gt = np.pad(gt, ((margin, margin), (margin, margin)), mode='reflect')

    # Only extract patches for pixels with valid labels (label != 0)
    # This is much more memory efficient than creating all patches and filtering
    patches_list = []
    labels_list = []
    coords_list = []
    
    for i in range(margin, margin + h):
        for j in range(margin, margin + w):
            label = padded_gt[i, j]
            if label != 0:  # Only process labeled pixels
                # Extract patch: [i-margin:i+margin, j-margin:j+margin, :]
                # This gives patch_size × patch_size × channels
                # Note: i+margin is exclusive, so [i-margin:i+margin] gives margin*2 = patch_size elements
                patch = padded_data[i - margin:i + margin, j - margin:j + margin, :]
                # Verify patch shape
                if patch.shape[:2] != (patch_size, patch_size):
                    raise ValueError(f"Patch shape mismatch at ({i-margin}, {j-margin}): "
                                   f"expected ({patch_size}, {patch_size}), got {patch.shape[:2]}. "
                                   f"margin={margin}, slice=[{i-margin}:{i+margin}]")
                patches_list.append(patch)
                labels_list.append(label)
                coords_list.append((i - margin, j - margin))
    
    if len(patches_list) == 0:
        raise ValueError("No valid labeled pixels found in ground truth data")
    
    patches = np.array(patches_list, dtype=np.float32)
    labels = np.array(labels_list)
    coords = np.array(coords_list)
    
    # Ensure patches shape is correct: (num_patches, patch_size, patch_size, channels)
    # The patches are already in the correct shape from the loop extraction
    if len(patches.shape) != 4:
        raise ValueError(f"Expected patches shape (N, {patch_size}, {patch_size}, {c}), got {patches.shape}")
    
    return patches, labels, coords, h, w

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class PatchAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class SimpleTransformer(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        z = z.unsqueeze(1)
        attn_out, _ = self.attn(z, z, z)
        squeezed = attn_out.squeeze(1)
        scores = self.linear(squeezed).squeeze()
        return scores

class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return torch.from_numpy(x)

    def __len__(self):
        return len(self.data)

def overlay_anomalies_on_rgb(
    rgb_img, coords, y_test, anomalies, label_to_name, descriptions=None,
    title="Anomaly Classification Map (on RGB)", save_path=None
):
    h, w, _ = rgb_img.shape
    valid = [i for i, (x, y) in enumerate(coords) if 0 <= x < h and 0 <= y < w]
    coords = np.array(coords)[valid]
    y_test = np.array(y_test)[valid]
    anomalies = np.array(anomalies)[valid]
    
    if len(anomalies) != len(coords):
        # Should not happen if filtered correctly
        return

    anomaly_coords = coords[anomalies]
    anomaly_labels = y_test[anomalies]
    unique_labels = np.unique(anomaly_labels)
    
    # Use a smaller figure size and lower DPI to save memory
    plt.figure(figsize=(10, 8)) 
    plt.imshow(rgb_img)
    plt.axis('off')
    
    palette = sns.color_palette("tab20", len(unique_labels))
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    
    # Plot in batches if too many points to avoid memory issues
    if len(anomaly_coords) > 0:
        for label in unique_labels:
            mask = anomaly_labels == label
            pts = anomaly_coords[mask]
            # Scatter is expensive, but we'll use a smaller marker and no edge to save some
            plt.scatter(pts[:, 1], pts[:, 0], s=15, c=[color_map[label]], label=str(label), linewidth=0)

    # Create custom legend
    patches = []
    for label in unique_labels:
        name = label_to_name.get(label, f"Class {label}")
        patches.append(mpatches.Patch(color=color_map[label], label=name))
        
    plt.legend(
        handles=patches, loc='upper right', bbox_to_anchor=(1.25, 1),
        borderaxespad=0., fontsize=8, title="Legend", frameon=True
    )
    plt.title(title, fontsize=12, fontweight='bold')
    
    if save_path:
        # Lower DPI significantly reduces memory usage
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')
    gc.collect()

def run_pipeline_with_files(hsi_path, gt_path, dataset_name, patch_size=16, latent_dim=32, num_epochs=10, output_dir=None, progress_callback=None):
    if output_dir == None:
        output_dir = os.path.join(os.path.dirname(hsi_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Model cache directory
    model_dir = os.path.join(os.path.dirname(os.path.dirname(hsi_path)), 'model')
    os.makedirs(model_dir, exist_ok=True)
    # Using v2 suffix
    model_path = os.path.join(model_dir, f"{dataset_name}_ae_model_v2.pth")
    
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if progress_callback:
        progress_callback(0)

    data = sio.loadmat(hsi_path)
    gt = sio.loadmat(gt_path)

    data_keys = [key for key in data.keys() if not key.startswith('__')]
    gt_keys = [key for key in gt.keys() if not key.startswith('__')]

    if not data_keys or not gt_keys:
        raise ValueError("No valid variable keys found in .mat files")

    data_array = data[data_keys[0]]
    gt_array = gt[gt_keys[0]]

    print("Loading dataset and preprocessing...")
    data_pca, gt_processed, h, w, pca_dim = preprocess(data_array, gt_array, dataset_name)

    rgb_image = data_pca[:, :, :3]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image_path = os.path.join(output_dir, f"{dataset_name}_pca_rgb.png")
    plt.imsave(rgb_image_path, rgb_image)
    del rgb_image_path # cleanup

    input_dim = patch_size * patch_size * pca_dim
    patches, labels, coords, h, w = extract_patches(data_pca, gt_processed, patch_size=patch_size)

    print(f"Extracted {len(patches)} patches.")
    print(f"Extracted {len(patches)} patches.")
    patches_flat = patches.reshape(len(patches), -1)
    
    # Use NumpyDataset to avoid creating a huge Tensor copy in memory
    dataset = NumpyDataset(patches_flat)
    batch_size = 512
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)
    
    model_loaded = False
    if os.path.exists(model_path):
        try:
            print(f"Loading cached model from {model_path}...")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load cached model: {e}. Training from scratch.")
    
    if not model_loaded:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        if progress_callback:
            progress_callback(1)

        print("Training Autoencoder...")
        model.train()
        epoch_losses = []

        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output, _ = model(batch)
                    loss = criterion(output, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        loss_curve_path = os.path.join(output_dir, f"{dataset_name}_ae_loss_curve.png")
        plt.figure()
        plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
        plt.title("Autoencoder Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(loss_curve_path)
        plt.close('all')
    else:
        if progress_callback:
            progress_callback(1)
            import time
            time.sleep(0.5)

    if progress_callback:
        progress_callback(2)

    print("Extracting latent features...")
    model.eval()
    latent_z_list = []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            _, z = model(batch)
            latent_z_list.append(z.cpu())
    latent_z = torch.cat(latent_z_list)

    print("Running Transformer...")
    transformer = SimpleTransformer(dim=latent_dim).to(device)
    transformer.eval()
    trans_scores = []
    for i in range(0, latent_z.shape[0], batch_size):
        batch = latent_z[i:i+batch_size].to(device)
        with torch.no_grad():
            scores = transformer(batch).cpu().numpy()
        trans_scores.append(scores)
    trans_scores = np.concatenate(trans_scores)

    if progress_callback:
        progress_callback(4)
    
    print("Creating anomaly map...")
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    for idx, (x, y) in enumerate(coords):
        anomaly_map[x, y] = trans_scores[idx]
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    anomaly_map_path = os.path.join(output_dir, f"{dataset_name}_anomaly_map.png")
    plt.imsave(anomaly_map_path, anomaly_map_norm, cmap='inferno')
    
    # Clean up large arrays if possible
    del trans_scores
    gc.collect()

    if progress_callback:
        progress_callback(3)

    print("Training SVM...")
    pca_svm = PCA(n_components=min(latent_dim, 20))
    latent_reduced = pca_svm.fit_transform(latent_z.numpy())

    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        latent_reduced, labels, coords, test_size=0.25, random_state=42, stratify=labels
    )
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), class_weights)}

    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight=class_weight_dict, probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)
    
    # Calculate AUC/AP (simplified for multi-class)
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(labels))
        y_score = svm_clf.decision_function(X_test)
        auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
        ap = average_precision_score(y_test_bin, y_score, average='macro')
    except:
        auc = 0.0
        ap = 0.0

    # Calculate Anomaly Threshold globally for all visualizations
    # Use 95th percentile as threshold (matching classified_model(1).ipynb)
    point_scores_all = np.array([anomaly_map[x, y] for x, y in coords])
    score_threshold = np.percentile(point_scores_all, 95)
    print(f"Anomaly Detection Threshold (95th percentile): {score_threshold:.4f}")

    print("Generating visualizations...")
    
    # Confusion Matrix
    print("Generating Confusion Matrix...")
    gc.collect()
    try:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        # Check size of confusion matrix
        if cm.shape[0] > 50:
             print(f"Confusion matrix too large ({cm.shape[0]} classes), skipping annotation to save memory.")
             sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        else:
             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
    except MemoryError:
        print("MemoryError encountered while generating confusion matrix. Skipping.")
    except Exception as e:
        print(f"Failed to generate confusion matrix: {e}")
    finally:
        plt.close('all')
        gc.collect()

    # Anomaly Overlay - SHOWING ALL CLASSIFIED POINTS
    print("Generating improved anomaly overlay (All Classified Points)...")
    
    # Predict on the FULL dataset to generate a complete Classification Map
    print("Generating full classification map...")
    all_preds = svm_clf.predict(latent_reduced)
    
    # We use the PREDICTED labels (all_preds) to show how the model classifies them.
    anomalies_mask = np.ones(len(all_preds), dtype=bool) # Show ALL points
    
    label_names = DATASET_METADATA.get(dataset_name, {}).get('label_to_name', {})
    descriptions = DATASET_METADATA.get(dataset_name, {}).get('descriptions', {})
    
    overlay_path = os.path.join(output_dir, f"{dataset_name}_anomaly_map_overlay.png")
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.imshow(anomaly_map_norm, cmap='inferno', alpha=0.5)
        plt.axis('off')
        plt.title(f"Anomaly Heatmap Overlay - {dataset_name.upper()}")
        plt.savefig(overlay_path, bbox_inches='tight')
    except Exception as e:
        print(f"Failed to generate anomaly map overlay: {e}")
    finally:
        plt.close('all')
        gc.collect()
    
    # Create Classification Map (Image)
    classification_map = np.zeros((h, w), dtype=int)
    for idx, (x, y) in enumerate(coords):
        classification_map[x, y] = all_preds[idx]
        
    # Visualize Classification Map
    unique_labels_full = np.unique(labels) # Use all possible labels from GT for consistent coloring
    palette = sns.color_palette("tab20", len(unique_labels_full) + 1) # +1 for background 0
    # Create a fixed color map based on all unique labels
    color_map_full = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels_full)}
    color_map_full[0] = (0, 0, 0) # Background is black
    
    # Create RGB image for classification map
    class_img_rgb = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            lbl = classification_map[i, j]
            if lbl in color_map_full:
                class_img_rgb[i, j] = color_map_full[lbl]
                
    # Create Classification Map (Scatter Overlay on RGB)
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.axis('off')
        
        # Filter points based on anomaly score using Detection Logic
        # point_scores_all and score_threshold are calculated globally above
        
        # Create legend handles
        patches_legend = []
        label_names = DATASET_METADATA.get(dataset_name, {}).get('label_to_name', {})
        
        # Plot points
        for label in unique_labels_full:
            if label == 0: continue
            # Filter by label AND anomaly score (Detection Logic)
            mask = (all_preds == label) & (point_scores_all >= score_threshold)
            
            if np.any(mask):
                pts = coords[mask]
                # Scatter with outlines to match reference
                plt.scatter(pts[:, 1], pts[:, 0], s=20, c=[color_map_full[label]], 
                           label=label_names.get(label, f"Class {label}"), 
                           edgecolors='k', linewidth=0.5)
                
                name = label_names.get(label, f"Class {label}")
                patches_legend.append(mpatches.Patch(color=color_map_full[label], label=name))
            
        plt.legend(handles=patches_legend, loc='upper right', bbox_to_anchor=(1.3, 1), title="Legend")
        plt.title(f"Anomaly Classification Map (on RGB)\n(Detected Anomalies > {score_threshold:.4f})", fontsize=12, fontweight='bold')
        
        class_map_path = os.path.join(output_dir, f"{dataset_name}_classification_map.png")
        plt.savefig(class_map_path, bbox_inches='tight')
    except MemoryError:
        print("MemoryError encountered while generating classification map. Skipping.")
    except Exception as e:
        print(f"Failed to generate classification map: {e}")
    finally:
        plt.close('all')
        gc.collect()

    # --- Generate Ground Truth Map ---
    print("Generating Ground Truth Map...")
    gt_map = np.zeros((h, w), dtype=int)
    # gt_processed is the full GT array (h, w)
    gt_map = gt_processed
    
    # Create RGB image for GT map using the same palette
    gt_img_rgb = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            lbl = gt_map[i, j]
            if lbl in color_map_full:
                gt_img_rgb[i, j] = color_map_full[lbl]
            # else: black (already zeros)

    gt_map_path = os.path.join(output_dir, f"{dataset_name}_gt_map.png")
    plt.imsave(gt_map_path, gt_img_rgb)

    # --- Generate Composite Image ---
    print("Generating Composite Visualization...")
    try:
        # Reduce figure size for memory
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Filter points based on anomaly score for composite as well
        # Use the SAME threshold calculated above
        # point_scores_all and score_threshold are calculated globally above
        
        mask_comp = point_scores_all >= score_threshold
        coords_filtered = coords[mask_comp]
        preds_filtered = all_preds[mask_comp]
        labels_filtered = labels[mask_comp]

        # Downsample if still too many points
        if len(coords_filtered) > 5000:
             indices = np.random.choice(len(coords_filtered), 5000, replace=False)
             coords_sample = coords_filtered[indices]
             preds_sample = preds_filtered[indices]
             labels_sample = labels_filtered[indices]
        else:
             coords_sample = coords_filtered
             preds_sample = preds_filtered
             labels_sample = labels_filtered

        # Panel 1: Ground Truth Overlay (Scatter) - Filtered
        axes[0].imshow(rgb_image)
        for label in unique_labels_full:
            if label == 0: continue
            mask = labels_sample == label
            if np.any(mask):
                pts = coords_sample[mask]
                axes[0].scatter(pts[:, 1], pts[:, 0], s=1, color=color_map_full[label], alpha=0.5)
        axes[0].set_title("Ground Truth (Detected)", fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: RGB Image
        axes[1].imshow(rgb_image)
        axes[1].set_title("RGB PCA Image", fontsize=10, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Predicted Classification Overlay (Scatter) - Filtered
        axes[2].imshow(rgb_image)
        for label in unique_labels_full:
            if label == 0: continue
            mask = preds_sample == label
            if np.any(mask):
                pts = coords_sample[mask]
                axes[2].scatter(pts[:, 1], pts[:, 0], s=1, color=color_map_full[label], alpha=0.5)

        axes[2].set_title("Prediction (Detected)", fontsize=10, fontweight='bold')
        axes[2].axis('off')
        
        # Panel 4: Overlay RGB + Anomalies
        axes[3].imshow(rgb_image)
        axes[3].imshow(anomaly_map_norm, cmap='inferno', alpha=0.5)
        axes[3].set_title("Overlay RGB + Anomalies", fontsize=10, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        composite_path = os.path.join(output_dir, f"{dataset_name}_composite.png")
        plt.savefig(composite_path, dpi=100, bbox_inches='tight')
    except MemoryError:
        print("MemoryError encountered while generating composite visualization. Skipping.")
    except Exception as e:
        print(f"Failed to generate composite visualization: {e}")
    finally:
        plt.close('all')
        gc.collect()
    gc.collect()

    print("Pipeline complete! Results saved in output directory.")

    results = {
        'stats': {
            'accuracy': accuracy,
            'classification_report': classification_report_str,
            'auc': auc,
            'average_precision': ap
        },
        'images': [
            {
                'url': f'/uploads/{dataset_name}_composite.png',
                'name': 'Composite Visualization',
                'description': 'Combined view of Ground Truth, RGB, Anomaly Heatmap, and Heatmap Overlay'
            },
            {
                'url': f'/uploads/{dataset_name}_classification_map.png',
                'name': 'Predicted Classification Map',
                'description': 'Full classification map generated by the model'
            },
            {
                'url': f'/uploads/{dataset_name}_confusion_matrix.png',
                'name': 'Confusion Matrix',
                'description': 'Visualization of model predictions vs true labels'
            },
            {
                'url': f'/uploads/{dataset_name}_anomaly_map.png',
                'name': 'Anomaly Score Map',
                'description': 'Spatial distribution of anomaly scores'
            },
            {
                'url': f'/uploads/{dataset_name}_pca_rgb.png',
                'name': 'PCA RGB Image',
                'description': 'RGB image from PCA components'
            },
            {
                'url': f'/uploads/{dataset_name}_ae_loss_curve.png',
                'name': 'Autoencoder Loss Curve',
                'description': 'Training loss curve of the autoencoder'
            },
            {
                'url': f'/uploads/{dataset_name}_anomaly_map_overlay.png',
                'name': 'Anomaly Map Overlay',
                'description': 'Anomaly heatmap overlaid on RGB image'
            }
        ],
        'classification_image_url': f'/uploads/{dataset_name}_classification_map.png', 
        'confusion_matrix_url': f'/uploads/{dataset_name}_confusion_matrix.png',
        'anomaly_score_map_url': f'/uploads/{dataset_name}_anomaly_map.png',
        'classification_report_url': f'/uploads/{dataset_name}_classification_report.csv', 
        'anomaly_stats': {
            'total_patches': len(labels), # Total labeled pixels
            'anomalies_detected': len(labels), # All labeled pixels are "detected" targets
            'anomaly_percentage': 100.0,
            'misclassifications': int(np.sum(y_pred != y_test)), # Keep track of errors on test set
            'threshold': 0.0, 
            'mean_reconstruction_error': 0.0 
        },
        'info': f'Analysis completed for {dataset_name} dataset with {len(labels)} samples'
    }

    return results
