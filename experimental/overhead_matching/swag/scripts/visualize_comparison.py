"""
Simple script to visualize and compare object detection results.
Run with: uv run python experimental/overhead_matching/swag/scripts/visualize_comparison.py
"""
import common.torch.load_torch_deps
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from experimental.overhead_matching.swag.model.spectral_landmark_extractor import SpectralLandmarkExtractor
from experimental.overhead_matching.swag.model.swag_config_types import SpectralLandmarkExtractorConfig
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput

# Load image
image_path = Path('/home/erick/Downloads/2008_000129.jpg')
image = tv.io.decode_image(str(image_path))
image = tv.transforms.v2.functional.to_dtype(image, scale=True)

print("Image loaded:", image.shape)

# Configure and create extractor
config = SpectralLandmarkExtractorConfig(
    dino_model="dinov3_vits16",
    feature_source="attention_keys",
    lambda_knn=0.0,
    max_landmarks_per_image=10,
    min_bbox_size=20,
    aggregation_method="weighted_mean"
)

extractor = SpectralLandmarkExtractor(config).cuda().eval()

# Run extraction
model_input = ModelInput(
    image=image.unsqueeze(0).cuda(),
    metadata=[{"path": image_path}]
)

with torch.no_grad():
    output = extractor(model_input)

# Get results
debug = output.debug
num_detected = (~output.mask[0]).sum().item()

print(f"Detected {num_detected} objects")
print(f"Eigenvalues: {debug['eigenvalues'][0][:num_detected].cpu()}")

# Create visualization
fig = plt.figure(figsize=(20, 12))

# Original image
ax = plt.subplot(3, 4, 1)
ax.imshow(image.permute(1, 2, 0).cpu())
ax.set_title('Original Image')
ax.axis('off')

# Image with bounding boxes
ax = plt.subplot(3, 4, 2)
ax.imshow(image.permute(1, 2, 0).cpu())
bboxes = debug['bboxes'][0].cpu().numpy()
colors = plt.cm.rainbow(np.linspace(0, 1, num_detected))

for i in range(num_detected):
    if not output.mask[0, i]:
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=colors[i], facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{i}', color=colors[i], fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))

ax.set_title(f'Detected Objects (n={num_detected})')
ax.axis('off')

# Show first 5 eigenvectors
eigenvectors = debug['eigenvectors'][0].cpu()
for i in range(min(5, eigenvectors.shape[0])):
    ax = plt.subplot(3, 4, 3 + i)
    eigvec = eigenvectors[i].numpy()
    im = ax.imshow(eigvec, cmap='viridis')
    eigenvalue = debug['eigenvalues'][0][i].cpu().item()
    ax.set_title(f'Eigenvector {i}\nλ={eigenvalue:.4f}', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

# Show object masks
if len(debug['object_masks'][0]) > 0:
    object_masks = debug['object_masks'][0].cpu()
    for i in range(min(6, object_masks.shape[0])):
        ax = plt.subplot(3, 4, 8 + i)
        mask = object_masks[i].numpy()
        ax.imshow(image.permute(1, 2, 0).cpu(), alpha=0.4)

        # Upsample mask to image size
        H, W = image.shape[1:]
        mask_upsampled = torch.nn.functional.interpolate(
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )[0, 0].numpy()

        im = ax.imshow(mask_upsampled, alpha=0.7, cmap='hot')
        ax.set_title(f'Object {i}', fontsize=9)
        ax.axis('off')

plt.tight_layout()
output_path = '/tmp/object_detection_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to {output_path}")
plt.show()
