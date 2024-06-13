import os
import glob
import shutil
import torch
import random
import torchvision
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Plot Transformed Images
def plot_transformed_images(image_paths, transform, n=3):
    # Create image list
    image_list = [f for f in glob.iglob(f"{image_paths}/*/*.jpg", recursive=True)] 
    # Select Random Images
    random_image_paths = random.sample(image_list, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2,figsize=(5.5,4))
            # Plot original image
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Plot images after transform
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            # Plot title
            fig.suptitle(f"Class: {os.path.basename(image_path).split('_')[0]}", fontsize=12)

# Training Step
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Test Step
def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    # Put model in eval mode
    model.eval() 
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred_logits = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Training Loop
def train_model(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, epochs: int, device:torch.device):
    
    # 1. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    # 2. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer,device=device)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn,device=device)
        # 3. Print results
        print(f"Epoch: {epoch+1} | "f"train_loss: {train_loss:.4f} | "f"train_acc: {train_acc:.4f} | "f"test_loss: {test_loss:.4f} | "f"test_acc: {test_acc:.4f}")
        # 4. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 5. Return the filled results at the end of the epochs
    return results

# Plot loss and accuracy
def plot_loss_curves(results: dict[str, list[float]]):
    # Get loss values
    loss = results['train_loss']
    test_loss = results['test_loss']
    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))
    # Setup a plot 
    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

# Predict custom image
def pred_and_plot_image(model: torch.nn.Module, image_path: str, class_names: list[str], device: torch.device, transform=None):    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    # 5. Turn on model evaluation mode and inference mode
    model.eval()

    # Turn on inference context manager
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu()*100:.2f}%"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

# Predict images
def pred_and_plot_images(model: torch.nn.Module, image_dir: str, class_names: list[str], 
                         device: torch.device, transform=None, num_images=3):

    # Get all image paths in the directory
    image_paths = [f for f in glob.iglob(f"{image_dir}/*.jpg", recursive=True)] 
    
    # Randomly sample image paths
    if len(image_paths) < num_images:
        print(f"Warning: Directory contains less than {num_images} images. Using {len(image_paths)} images.")
        num_images = len(image_paths)
    sampled_image_paths = random.sample(image_paths, num_images)

    # Loop through sampled images and predict/plot
    for image_path in sampled_image_paths:
        print(f"Predicting for image: {image_path}")
        pred_and_plot_image(model, image_path, class_names, device, transform)

# Check for images 
def is_image(filename):
  extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
  return any(filename.lower().endswith(ext) for ext in extensions)

# Delete Unwanted Images
def delete_images(root_dir, max_files=115):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print('Das ist ',dirpath)
        count = 0
        for filename in glob.iglob(os.path.join(dirpath, "*")):
            if is_image(filename) and count < max_files:
                os.remove(filename)
                count += 1
                print(f"Deleted {filename}")

    if count == 0:
        print("No images found for deletion.")
    else:
        print(f"Successfully deleted {count} images.")

# Skip some images then delete the rest
def delete_images_after_skip(root_dir, max_files_to_skip=115):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        count = 0
        skipped = 0  # Track the number of skipped images
        for filename in glob.iglob(os.path.join(dirpath, "*")):
            if is_image(filename):
                if skipped < max_files_to_skip:
                    skipped += 1
                    continue  # Skip the first max_files_to_skip images
                os.remove(filename)
                count += 1
                print(f"Deleted {filename}")

    if count == 0:
        print("No images found for deletion after skipping the first", max_files_to_skip)
    else:
        print(f"Successfully deleted {count} images (after skipping the first {max_files_to_skip}).")

# Move Images
def move_and_rename_images(root_dir, target_dir, max_files=5):
  for dirpath, dirnames, filenames in os.walk(root_dir):
    subdir_name = os.path.basename(dirpath)  # Get subdirectory name
    count = 0  # Track total moved images
    for filename in glob.iglob(os.path.join(dirpath, "*")):
      if is_image(filename) and count < max_files:
        new_filename = f"{subdir_name}_{count}{os.path.splitext(filename)[1]}"
        target_path = os.path.join(target_dir, new_filename)
        shutil.move(filename, target_path)  # Move the file
        count += 1
        print(f"Moved {filename} to {target_path}")

  if count == 0:
    print("No images found for moving.")
  else:
    print(f"Successfully moved {count} images.")

    