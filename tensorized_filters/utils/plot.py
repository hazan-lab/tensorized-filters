# =============================================================================#
# Authors: Windsor Nguyen
# File: plot.py
# =============================================================================#

"""A malleable plotting script."""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from tensorized_filters.utils.logger import logger
from scipy.ndimage import gaussian_filter1d


def load_data(file_path, data_type=None):
    """
    Loads loss data from a file, optionally filtering by data type ('train', 'val', etc.).

    Args:
        file_path (str): Path to the data file.
        data_type (str, optional): Type of data to load ('train', 'val', etc.). Defaults to None.

    Returns:
        np.ndarray: Array of loss values.
    """
    while True:
        try:
            with open(file_path, "r") as f:
                data = []
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # If data_type is specified, filter by it
                            if data_type and parts[1].lower() != data_type.lower():
                                continue  # Skip lines that don't match the desired type

                            # Attempt to convert the third element to float
                            try:
                                loss = float(parts[2])
                                data.append(loss)
                            except ValueError as e:
                                logger.error(f"Cannot convert '{parts[2]}' to float in line: {line.strip()}")
                                raise ValueError(f"Invalid loss value in line: {line.strip()}") from e
                        else:
                            logger.error(f"Line has insufficient parts: {line.strip()}")
                            raise ValueError(f"Invalid format in line: {line.strip()}")
            data = np.array(data)
            if data_type:
                logger.info(f"Loaded {len(data)} '{data_type}' data points from {file_path}.")
            else:
                logger.info(f"Loaded {len(data)} data points from {file_path}.")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            file_path = prompt_valid_file_path(f"Please enter a valid file path for '{file_path}': ")
        except ValueError as e:
            logger.error(f"Invalid data format in file: {file_path} - {e}")
            file_path = prompt_valid_file_path(f"Please enter a valid file path for '{file_path}': ")


def prompt_valid_file_path(prompt_message):
    """
    Prompts the user to enter a valid file path until a valid one is provided.

    Args:
        prompt_message (str): The prompt message to display to the user.

    Returns:
        str: A valid file path entered by the user.
    """
    while True:
        user_input = input(prompt_message).strip('"').strip("'")
        if os.path.isfile(user_input):
            return user_input
        else:
            logger.warning(f"Invalid file path: {user_input}. Please try again.")


def apply_gaussian_smoothing(data: np.ndarray, sigma: float = 2) -> np.ndarray:
    """
    Applies Gaussian smoothing to the data.

    Args:
        data (np.ndarray): The data to smooth.
        sigma (float, optional): The sigma parameter for Gaussian kernel. Defaults to 2.

    Returns:
        np.ndarray: Smoothed data.
    """
    return gaussian_filter1d(data, sigma)


def plot_data(
    data_list: list[np.ndarray],
    time_steps_list: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes = None,
    smoothing: bool = True,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the provided data.

    Args:
        data_list (list[np.ndarray]): List of data arrays to plot.
        time_steps_list (list[np.ndarray]): Corresponding list of time steps for each data array.
        labels (list[str]): Labels for each plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        ax (plt.Axes, optional): Matplotlib Axes object. Creates one if None. Defaults to None.
        smoothing (bool, optional): Whether to apply Gaussian smoothing. Defaults to True.
        xlim (tuple[float, float], optional): X-axis limits. Defaults to None.
        ylim (tuple[float, float], optional): Y-axis limits. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for data, time_steps, label in zip(data_list, time_steps_list, labels, strict=True):
        if smoothing:
            data = apply_gaussian_smoothing(data)
        ax.plot(time_steps, data, linewidth=2, label=label)

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12, loc="best", frameon=True, fancybox=True, shadow=True)

    # Apply x and y limits if specified
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Improve aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    fig.tight_layout()
    return fig, ax


def get_user_input(prompt, data_type=str, valid_options=None):
    """
    Prompts the user for input and attempts to convert it to the specified data type.
    Optionally validates the input against a list of valid options.

    Args:
        prompt (str): The input prompt.
        data_type (type, optional): The desired data type. Defaults to str.
        valid_options (list, optional): A list of valid inputs. Defaults to None.

    Returns:
        Any: The user input converted to the specified data type.
    """
    while True:
        user_input = input(prompt).strip()
        try:
            converted_input = data_type(user_input)
            if valid_options:
                if converted_input.lower() not in valid_options:
                    logger.warning(f"Invalid option: {converted_input}. Valid options are: {valid_options}")
                    continue
            return converted_input
        except ValueError:
            logger.warning("Invalid input type. Please try again.")


def get_axis_range(axis_name: str):
    """
    Prompts the user to enter the minimum and maximum values for a specified axis.

    Args:
        axis_name (str): The name of the axis ('x' or 'y').

    Returns:
        tuple[float, float]: The (min, max) range for the axis.
    """
    while True:
        try:
            min_val = float(input(f"Enter the minimum value for the {axis_name}-axis: "))
            max_val = float(input(f"Enter the maximum value for the {axis_name}-axis: "))
            if min_val >= max_val:
                logger.warning(f"Minimum {axis_name}-axis value must be less than the maximum. Please try again.")
                continue
            return (min_val, max_val)
        except ValueError:
            logger.warning("Invalid input. Please enter numerical values.")


def validate_data_type(data_type, allowed_types):
    """
    Validates if the provided data_type is within the allowed_types.

    Args:
        data_type (str): The data type to validate.
        allowed_types (list): List of allowed data types.

    Returns:
        bool: True if valid, False otherwise.
    """
    return data_type.lower() in allowed_types


def prompt_data_type(plot_number=None):
    """
    Prompts the user to enter a valid data type.

    Args:
        plot_number (int, optional): The plot number for reference. Defaults to None.

    Returns:
        str: A valid data type entered by the user.
    """
    allowed_types = ["train", "val", "test", "grad_norm", "other"]
    prompt_msg = f"Enter the data type for plot {plot_number} (options: {', '.join(allowed_types)}): "
    while True:
        data_type = input(prompt_msg).strip().lower()
        if data_type in allowed_types:
            return data_type
        else:
            logger.warning(f"Invalid data type: '{data_type}'. Allowed types are: {allowed_types}")


def main():
    logger.info("Welcome to the Hazan Lab enhanced plotting script!")

    plot_data_list = []
    plot_time_steps_list = []
    plot_labels = []
    plot_titles = []
    plot_xlabels = []
    plot_ylabels = []
    plot_xranges = []
    plot_yranges = []

    while True:
        logger.info("\nWhat do you want to plot today?")
        logger.info("1. Train losses")
        logger.info("2. Validation losses")
        logger.info("3. Test losses")
        logger.info("4. Train and val")
        logger.info("5. Gradient norms")
        logger.info("6. Other files together")

        choice = get_user_input(
            "Enter your choice (1-6): ", data_type=str, valid_options=["1", "2", "3", "4", "5", "6"]
        )

        if choice in ["1", "2", "3", "5", "6"]:
            num_plots = get_user_input("Enter the number of plots: ", data_type=int)
            if num_plots <= 0:
                logger.warning("Number of plots must be at least 1. Please try again.")
                continue

            data_list = []
            time_steps_list = []
            labels = []

            for i in range(1, num_plots + 1):
                logger.info(f"\nConfiguring plot {i} of {num_plots}.")

                file_path = prompt_valid_file_path(f"Enter the file path for plot {i} (.txt): ")

                # Determine data_type based on choice
                if choice == "1":
                    data_type = "train"
                elif choice == "2":
                    data_type = "val"
                elif choice == "3":
                    data_type = "test"
                elif choice == "5":
                    data_type = "grad_norm"
                elif choice == "6":
                    data_type = prompt_data_type(plot_number=i)

                data = load_data(file_path, data_type=data_type if choice != "6" else data_type)
                if data.size == 0:
                    logger.warning(f"No data loaded for plot {i}. Skipping this plot.")
                    continue
                data_list.append(data)

                label = input(f"Enter the legend name for plot {i}: ").strip()
                if not label:
                    label = f"Plot {i}"
                labels.append(label)

                # Handle time_steps
                if choice == "2" and data_type == "val":
                    use_external_ts = get_user_input(
                        f"Do you want to provide a separate time steps file for plot {i}? (y/n): ",
                        data_type=str,
                        valid_options=["y", "n"],
                    ).lower()

                    if use_external_ts == "y":
                        time_steps_file_path = prompt_valid_file_path(
                            f"Enter the file path for validation time steps plot {i} (.txt): "
                        )
                        time_steps_data = load_data(time_steps_file_path, data_type="val")
                        if time_steps_data.size > 0:
                            time_steps_list.append(time_steps_data)
                        else:
                            logger.warning(
                                f"Validation time steps data not found or empty for plot {i}. Using default steps."
                            )
                            time_steps_list.append(np.arange(len(data)))
                    else:
                        time_steps_list.append(np.arange(len(data)))
                elif choice == "6":
                    # For 'Other files together', determine if time_steps are needed based on data_type
                    # Assuming 'val' type may require time_steps, else default
                    if data_type == "val":
                        use_external_ts = get_user_input(
                            f"Do you want to provide a separate time steps file for plot {i}? (y/n): ",
                            data_type=str,
                            valid_options=["y", "n"],
                        ).lower()

                        if use_external_ts == "y":
                            time_steps_file_path = prompt_valid_file_path(
                                f"Enter the file path for validation time steps plot {i} (.txt): "
                            )
                            time_steps_data = load_data(time_steps_file_path, data_type="val")
                            if time_steps_data.size > 0:
                                time_steps_list.append(time_steps_data)
                            else:
                                logger.warning(
                                    f"Validation time steps data not found or empty for plot {i}. Using default steps."
                                )
                                time_steps_list.append(np.arange(len(data)))
                        else:
                            time_steps_list.append(np.arange(len(data)))
                    else:
                        # For other data_types, use default steps
                        time_steps_list.append(np.arange(len(data)))
                else:
                    # For other choices, use default steps
                    time_steps_list.append(np.arange(len(data)))

            title = input("Enter the title for the plot: ").strip()
            if not title:
                # Provide default titles based on choice
                default_titles = {
                    "1": "Training Losses",
                    "2": "Validation Losses",
                    "3": "Test Losses",
                    "5": "Gradient Norms",
                    "6": "Custom Plot",
                }
                title = default_titles.get(choice, "Plot")

            if choice == "1":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title)
                plot_xlabels.append("Epochs")
                plot_ylabels.append("Loss")
            elif choice == "2":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title)
                plot_xlabels.append("Steps")
                plot_ylabels.append("Loss")
            elif choice == "3":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title)
                plot_xlabels.append("Epochs")
                plot_ylabels.append("Loss")
            elif choice == "5":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title)
                plot_xlabels.append("Steps")
                plot_ylabels.append("Norm")
            elif choice == "6":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title)
                # Prompt user for x and y labels for custom plots
                x_label = input("Enter x-axis label: ").strip()
                y_label = input("Enter y-axis label: ").strip()
                plot_xlabels.append(x_label if x_label else "X-axis")
                plot_ylabels.append(y_label if y_label else "Y-axis")

            # Initialize x and y ranges for each plot as None
            plot_xranges.append(None)
            plot_yranges.append(None)

        elif choice == "4":
            # Plot both Train and Val together
            logger.info("\nConfiguring combined Train and Validation plots.")

            # Configure Train Plots
            num_train_plots = get_user_input("Enter the number of train plots: ", data_type=int)
            if num_train_plots <= 0:
                logger.warning("Number of train plots must be at least 1. Please try again.")
                continue

            train_data_list = []
            train_time_steps_list = []
            train_labels = []
            for i in range(1, num_train_plots + 1):
                logger.info(f"\nConfiguring Train plot {i} of {num_train_plots}.")

                file_path = prompt_valid_file_path(f"Enter the file path for train plot {i} (.txt): ")

                data = load_data(file_path, data_type="train")
                if data.size == 0:
                    logger.warning(f"No train data loaded for plot {i}. Skipping this plot.")
                    continue
                train_data_list.append(data)

                label = input(f"Enter the legend name for train plot {i}: ").strip()
                if not label:
                    label = f"Train Plot {i}"
                train_labels.append(label)

                train_time_steps_list.append(np.arange(len(data)))

            # Configure Validation Plots
            num_val_plots = get_user_input("Enter the number of validation plots: ", data_type=int)
            if num_val_plots <= 0:
                logger.warning("Number of validation plots must be at least 1. Please try again.")
                continue

            val_data_list = []
            val_time_steps_list = []
            val_labels = []
            for i in range(1, num_val_plots + 1):
                logger.info(f"\nConfiguring Validation plot {i} of {num_val_plots}.")

                file_path = prompt_valid_file_path(f"Enter the file path for val plot {i} (.txt): ")

                data = load_data(file_path, data_type="val")
                if data.size == 0:
                    logger.warning(f"No validation data loaded for plot {i}. Skipping this plot.")
                    continue
                val_data_list.append(data)

                label = input(f"Enter the legend name for val plot {i}: ").strip()
                if not label:
                    label = f"Val Plot {i}"
                val_labels.append(label)

                # Handle time_steps for validation plots
                use_external_ts = get_user_input(
                    f"Do you want to provide a separate time steps file for val plot {i}? (y/n): ",
                    data_type=str,
                    valid_options=["y", "n"],
                ).lower()

                if use_external_ts == "y":
                    time_steps_file_path = prompt_valid_file_path(
                        f"Enter the file path for validation time steps plot {i} (.txt): "
                    )
                    time_steps_data = load_data(time_steps_file_path, data_type="val")
                    if time_steps_data.size > 0:
                        val_time_steps_list.append(time_steps_data)
                    else:
                        logger.warning(
                            f"Validation time steps data not found or empty for plot {i}. Using default steps."
                        )
                        val_time_steps_list.append(np.arange(len(data)))
                else:
                    val_time_steps_list.append(np.arange(len(data)))

            combined_data = train_data_list + val_data_list
            combined_time_steps = train_time_steps_list + val_time_steps_list
            combined_labels = train_labels + val_labels

            plot_data_list.append(combined_data)
            plot_time_steps_list.append(combined_time_steps)
            plot_labels.append(combined_labels)
            title = input("Enter the title for the training and validation plot: ").strip()
            plot_titles.append(title if title else "Training and Validation Losses")
            plot_xlabels.append("Steps")
            plot_ylabels.append("Loss")

            # Initialize x and y ranges for this plot as None
            plot_xranges.append(None)
            plot_yranges.append(None)

        else:
            logger.warning("Invalid choice. Please try again.")
            continue

        more_plots = get_user_input(
            "Do you want to add more plots? (y/n): ", data_type=str, valid_options=["y", "n"]
        ).lower()
        if more_plots != "y":
            break

    # Ask if the user wants to specify axis ranges
    specify_xrange = (
        get_user_input(
            "Do you want to specify the x-axis range? (y/n): ", data_type=str, valid_options=["y", "n"]
        ).lower()
        == "y"
    )
    if specify_xrange:
        x_range = get_axis_range("x")
    else:
        x_range = None

    specify_yrange = (
        get_user_input(
            "Do you want to specify the y-axis range? (y/n): ", data_type=str, valid_options=["y", "n"]
        ).lower()
        == "y"
    )
    if specify_yrange:
        y_range = get_axis_range("y")
    else:
        y_range = None

    smoothing_input = get_user_input(
        "Do you want to apply Gaussian smoothing? (y/n): ", data_type=str, valid_options=["y", "n"]
    ).lower()
    smoothing = smoothing_input == "y"

    while True:
        save_path = input("Which directory should the plot be saved in?: ").strip('"').strip("'")
        if not save_path:
            logger.warning("Save path cannot be empty. Please try again.")
            continue
        try:
            os.makedirs(save_path, exist_ok=True)
            break
        except Exception as e:
            logger.error(f"Failed to create/access directory '{save_path}': {e}")
            logger.info("Please enter a different directory path.")

    same_plot = get_user_input(
        "Do you want to plot all graphs on the same plot? (y/n): ", data_type=str, valid_options=["y", "n"]
    ).lower()

    if same_plot == "y":
        fig, ax = plt.subplots(figsize=(12, 8))
        for data_list, time_steps_list, labels, title, xlabel, ylabel in zip(
            plot_data_list,
            plot_time_steps_list,
            plot_labels,
            plot_titles,
            plot_xlabels,
            plot_ylabels,
            strict=True,
        ):
            plot_data(
                data_list,
                time_steps_list,
                labels,
                title,
                xlabel,
                ylabel,
                ax,
                smoothing,
                xlim=x_range,
                ylim=y_range,
            )

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"combined_plot_{current_time}.png"
        while True:
            filename = input(f"Enter the file name for the combined plot (default: {default_filename}): ").strip()
            if not filename:
                filename = default_filename
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
                logger.warning("Invalid file extension. Please use one of: .png, .jpg, .jpeg, .pdf, .svg")
                continue
            try:
                plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
                logger.info(f"Combined plot saved at: {os.path.join(save_path, filename)}")
                break
            except Exception as e:
                logger.error(f"Failed to save plot '{filename}': {e}")
                logger.info("Please enter a different filename.")

    else:
        for i, (data_list, time_steps_list, labels, title, xlabel, ylabel) in enumerate(
            zip(
                plot_data_list,
                plot_time_steps_list,
                plot_labels,
                plot_titles,
                plot_xlabels,
                plot_ylabels,
                strict=True,
            )
        ):
            fig, ax = plot_data(
                data_list,
                time_steps_list,
                labels,
                title,
                xlabel,
                ylabel,
                smoothing=smoothing,
                xlim=x_range,
                ylim=y_range,
            )
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_filename = f"plot_{i+1}_{current_time}.png"
            while True:
                filename = input(f"Enter the filename for plot {i+1} (default: {default_filename}): ").strip()
                if not filename:
                    filename = default_filename
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
                    logger.warning("Invalid file extension. Please use one of: .png, .jpg, .jpeg, .pdf, .svg")
                    continue
                try:
                    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
                    logger.info(f"Plot {i+1} saved at: {os.path.join(save_path, filename)}")
                    break
                except Exception as e:
                    logger.error(f"Failed to save plot '{filename}': {e}")
                    logger.info("Please enter a different filename.")

    plt.show()
    logger.info("Plotting completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Plotting interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
