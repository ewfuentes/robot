import common.torch.load_torch_deps
from experimental.learn_descriptors.depth_models import UniDepthV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

model = UniDepthV2('large', device='cuda:0')

samples = [
    ('/data/overhead_matching/datasets/pinhole_images/Framingham/1114487373893102,42.321599,-71.343132,/yaw_000.jpg', 'Framingham_yaw000'),
    ('/data/overhead_matching/datasets/pinhole_images/Framingham/1114487373893102,42.321599,-71.343132,/yaw_090.jpg', 'Framingham_yaw090'),
    ('/data/overhead_matching/datasets/pinhole_images/nightdrive/045-F-NBRguoXKTawS5H,42.368794,-71.109684,/yaw_000.jpg', 'nightdrive_yaw000'),
    ('/data/overhead_matching/datasets/pinhole_images/boston_snowy/MZqd2lRBI2cbvWOHaxDRls,42.361994,-71.080753,/yaw_000.jpg', 'boston_snowy_yaw000'),
]

for img_path, label in samples:
    p = Path(img_path)
    if not p.exists():
        print(f'SKIP: {p}')
        continue

    depth = model.infer(p)
    depth = np.asarray(depth, dtype=np.float32).squeeze()

    img = Image.open(p).resize((512, 512))
    depth_img = Image.fromarray(depth).resize((512, 512))
    depth_resized = np.array(depth_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(img)
    ax1.set_title(f'{label} - RGB')
    ax1.axis('off')

    im = ax2.imshow(depth_resized, cmap='turbo')
    ax2.set_title(f'Depth (min={depth.min():.1f}m, max={depth.max():.1f}m)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='meters')

    plt.tight_layout()
    out = f'/tmp/depth_example_{label}.png'
    plt.savefig(out, dpi=100)
    plt.close()
    print(f'Saved {out}')
