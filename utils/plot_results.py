import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch


def plot_classification_results(
    dataloader, model, cnn, noise_scheduler, device, num_examples=12, correct=True
):
    """
    Plots classification results (correct or incorrect) from a model and dataloader.

    Args:
        dataloader (DataLoader): Dataloader for validation/test data.
        model (nn.Module): Classifier model to test.
        cnn (nn.Module): Pretrained CNN for feature extraction.
        noise_scheduler: Noise scheduler used in the model.
        device (torch.device): Device to run the model on.
        num_examples (int): Number of examples to plot.
        correct (bool): If True, plot correctly classified examples; otherwise, plot incorrect ones.
    """
    num_cols = 4
    num_rows = (num_examples + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(12, 5 * num_rows))
    gs = GridSpec(num_rows * 2, num_cols, hspace=0.4)

    plotted_count = 0  # Counter for plotted samples

    with torch.no_grad():
        for x, y in dataloader:
            if plotted_count >= num_examples:
                break  # Stop if we've plotted enough images

            x, y = x.to(device), y.to(device)
            features = cnn.extract_features(x)
            timesteps = (
                torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (y.shape[0],)
                )
                .long()
                .to(device)
            )
            pred_noise = model(features=features, labels=y, timesteps=timesteps)

            # Add noise to labels
            noisy_labels = noise_scheduler.add_noise(y, pred_noise, timesteps)

            # Perform forward pass
            predicted_eps = model(
                features=features, labels=noisy_labels, timesteps=timesteps
            )

            # Apply threshold for binary classification
            y_pred_binary = (predicted_eps > 0.5).float()

            # Filter based on correctness
            match_condition = (y_pred_binary == y).all(dim=1)
            samples_to_plot = match_condition if correct else ~match_condition

            if samples_to_plot.any():
                sample_indices = torch.where(samples_to_plot)[0]

                for idx in sample_indices:
                    if plotted_count >= num_examples:
                        break  # Stop if we've plotted enough images

                    # Get the sample data
                    img = x[idx].cpu().squeeze()
                    probs = predicted_eps[idx].cpu().detach().numpy()

                    # Row and column index for plotting
                    row = plotted_count // num_cols
                    col = plotted_count % num_cols

                    # Plot the image
                    ax_image = fig.add_subplot(gs[row * 2, col])
                    ax_image.imshow(img, cmap="gray")
                    ax_image.axis("off")
                    ax_image.set_title(
                        "Correct" if correct else "Incorrect", fontsize=8
                    )

                    # Plot the probabilities
                    ax_probs = fig.add_subplot(gs[row * 2 + 1, col])
                    ax_probs.bar(
                        range(len(probs)), probs, color="blue" if correct else "red"
                    )
                    ax_probs.set_xticks(range(len(probs)))
                    ax_probs.set_xticklabels(range(len(probs)), fontsize=8)
                    ax_probs.set_ylim(0, 1)
                    ax_probs.set_title("Probabilities", fontsize=8)

                    plotted_count += 1  # Increment the counter for plotted examples

    plt.tight_layout()
    plt.show()
