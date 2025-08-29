import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import matplotlib.patches as patches
import torch
import os

def visualize_prediction(model, dataset, device, config, idx=0, score_threshold=0.5, title='Prediction', save_path=None):
    model.eval()

    image, target = dataset[idx]
    image_tensor = image.to(device).unsqueeze(0)
    output = model(image_tensor)[0]

    boxes = output['boxes'].cpu()
    scores = output['scores'].cpu()
    labels = output['labels'].cpu()

    img = F.to_pil_image(image)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)


    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        cls_name = config.CLASSES[label -1] if 1 <= label <= len(config.CLASSES) else f"class {label}"
        ax.text(x1, y1-5, f"{cls_name} {score:.2f}", color='red', fontsize=12)

    ax.set_title(title)
    ax.axis('off')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"result_{idx}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"[Saved] Visualization saved to {filename}")
    plt.show()

def compare_predictions(models, model_names, dataset, device, config, idx=0, score_threshold=0.5, save_path=None):
    image, _ = dataset[idx]
    image_tensor = image.to(device).unsqueeze(0)
    img = F.to_pil_image(image)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (model, name) in enumerate(zip(models, model_names)):
        model.eval()
        output = model(image_tensor)[0]
        boxes = output['boxes'].cpu()
        scores = output['scores'].cpu()
        labels = output['labels'].cpu()

        axes[i].imshow(img)
        for box, score, label in zip(boxes, scores, labels):
            if score < score_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            cls_name = config.CLASSES[label -1] if 1 <= label <= len(config.CLASSES) else f"class {label}"
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')

            axes[i].add_patch(rect)
            axes[i].text(x1, y1 -5, f"{cls_name} {score:.2f}", color='red', fontsize=10)

        axes[i].set_title(name)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_ground_truth_and_predictions(models, model_names, dataset, device, config, target_class=None, start_idx = 0, score_threshold=0.75, save_path=None):
    """
    Visualize both the ground truth and predictions from multiple models for a sample
    containing a specified target class.
    
    Args:
        models (list[nn.Module]): List of object detection models to compare.
        model_names (list[str]): Names for each model 
        dataset (Dataset): Dataset object returning (image, target) pairs.
        target_class: Class name or index to search for.
        start_idx: Index to start searching for a sample containing the target class.
        score_threshold: Score threshold to draw predictions.
    
    """
    # Resolve class index
    class_idx = None
    if target_class is not None:
        if isinstance(target_class, str):
            class_idx = config.CLASSES.index(target_class)
        else:
            class_idx = target_class

    # Find image that contains the class
    found = False
    for idx in range(start_idx, len(dataset)):
        data = dataset[idx]
        if data is None:
            continue
        image, target = data

        if class_idx is None or class_idx in target['labels'].tolist():
            found = True
            break


    if not found:
        print(f"[Warning] Class '{target_class}' not found in dataset starting from index {start_idx}")
        return
    
    # Visualization
    image, target = dataset[idx]
    image_tensor = image.to(device).unsqueeze(0)
    img = F.to_pil_image(image)

    n = len(models) + 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    axes[0].imshow(img)
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.tolist()
        cls_name = config.CLASSES[label] if 1 <= label <= len(config.CLASSES) else f"class {label}"
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, f"{cls_name} (GT)", color='green', fontsize=10)

    axes[0].set_title("Ground Truth")
    axes[0].axis('off')


    for i, (model, name) in enumerate(zip(models, model_names), start=1):
            model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)[0]

            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu()

            axes[i].imshow(img)
            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                cls_name = config.CLASSES[label] if 1 <= label <= len(config.CLASSES) else f"class {label}"
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor='red', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].text(x1, y1 - 5, f"{cls_name} {score:.2f}", color='red', fontsize=10)

            axes[i].set_title(name)
            axes[i].axis('off')

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"compare_{idx}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"[Saved] Visualization saved to {filename}")
    plt.show()