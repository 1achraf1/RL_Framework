import numpy as np
import matplotlib.pyplot as plt

def plot_policy(env, Q):
    """
    Visualize the learned policy as arrows on the GridWorld.
    
    Parameters
    ----------
    env : GridWorld
        The environment instance.
    Q : np.ndarray
        Q-table of shape (height, width, 4).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # === Policy arrows ===
    ax1 = axes[0]
    grid = np.zeros((env.height, env.width))

    # Mark obstacles and goals
    for obs in env.obstacles:
        grid[obs] = -1
    for goal in env.goals:
        grid[goal] = 1

    ax1.imshow(grid, cmap="RdYlGn", alpha=0.3,
               extent=[0, env.width, env.height, 0])
    ax1.set_xticks(np.arange(0, env.width + 1, 1))
    ax1.set_yticks(np.arange(0, env.height + 1, 1))
    ax1.grid(color="black", linewidth=1)
    ax1.set_title("Learned Policy (Arrows)", fontsize=14, fontweight="bold")

    # Arrow directions: 0=up, 1=right, 2=down, 3=left
    arrow_dx = [0, 0.3, 0, -0.3]
    arrow_dy = [-0.3, 0, 0.3, 0]

    for r in range(env.height):
        for c in range(env.width):
            if (r, c) not in env.obstacles and (r, c) not in env.goals:
                best_action = np.argmax(Q[r, c])
                ax1.arrow(c + 0.5, r + 0.5,
                          arrow_dx[best_action], arrow_dy[best_action],
                          head_width=0.2, head_length=0.15,
                          fc="blue", ec="blue", linewidth=2)
            elif (r, c) in env.obstacles:
                ax1.text(c + 0.5, r + 0.5, "X", ha="center", va="center",
                         fontsize=20, fontweight="bold", color="red")
            elif (r, c) in env.goals:
                ax1.text(c + 0.5, r + 0.5, "G", ha="center", va="center",
                         fontsize=20, fontweight="bold", color="green")

    # === Value function ===
    ax2 = axes[1]
    value_grid = np.max(Q, axis=2).copy()

    # Set obstacles and goals to NaN for cleaner display
    for obs in env.obstacles:
        value_grid[obs] = np.nan

    im = ax2.imshow(value_grid, cmap="viridis",
                    extent=[0, env.width, env.height, 0])
    ax2.set_xticks(np.arange(0, env.width + 1, 1))
    ax2.set_yticks(np.arange(0, env.height + 1, 1))
    ax2.grid(color="white", linewidth=1)
    ax2.set_title("State Values (Max Q-value)", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Value", rotation=270, labelpad=20)

    # Add text annotations
    for r in range(env.height):
        for c in range(env.width):
            if (r, c) in env.obstacles:
                ax2.text(c + 0.5, r + 0.5, "X", ha="center", va="center",
                         fontsize=16, fontweight="bold", color="red")
            elif (r, c) in env.goals:
                ax2.text(c + 0.5, r + 0.5, "G", ha="center", va="center",
                         fontsize=16, fontweight="bold", color="white")
            else:
                max_val = np.max(Q[r, c])
                ax2.text(c + 0.5, r + 0.5, f"{max_val:.1f}",
                         ha="center", va="center",
                         fontsize=10, color="white", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_q_values_heatmap(env, Q):
    """
    Plot Q-values for each action as separate heatmaps.
    
    Parameters
    ----------
    env : GridWorld
        The environment instance.
    Q : np.ndarray
        Q-table of shape (height, width, 4).
    """
    action_names = ["Up", "Right", "Down", "Left"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for action_idx in range(4):
        ax = axes[action_idx]
        q_grid = Q[:, :, action_idx].copy()

        # Set obstacles to NaN
        for obs in env.obstacles:
            q_grid[obs] = np.nan

        im = ax.imshow(q_grid, cmap="RdYlGn",
                       extent=[0, env.width, env.height, 0])
        ax.set_xticks(np.arange(0, env.width + 1, 1))
        ax.set_yticks(np.arange(0, env.height + 1, 1))
        ax.grid(color="black", linewidth=1)
        ax.set_title(f"Q-values for Action: {action_names[action_idx]}",
                     fontsize=12, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Q-value", rotation=270, labelpad=15)

        # Add text annotations
        for r in range(env.height):
            for c in range(env.width):
                if (r, c) in env.obstacles:
                    ax.text(c + 0.5, r + 0.5, "X", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="black")
                elif (r, c) in env.goals:
                    ax.text(c + 0.5, r + 0.5, "G", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="blue")
                else:
                    q_val = Q[r, c, action_idx]
                    ax.text(c + 0.5, r + 0.5, f"{q_val:.1f}",
