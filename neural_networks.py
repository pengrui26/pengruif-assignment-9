import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle, FancyArrowPatch

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X  # Store input
        self.Z1 = np.dot(X, self.W1) + self.b1  # Pre-activation of hidden layer
        # Apply activation function to Z1 to get A1
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))
        else:
            raise ValueError("Unsupported activation function")
        # Compute output layer pre-activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # Apply sigmoid activation to output layer
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def backward(self, X, y):
        m = y.shape[0]
        # Output layer gradients
        dZ2 = self.A2 - y  # derivative of loss w.r.t. Z2
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        # Compute dZ1 depending on activation function
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.power(self.A1, 2))
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            dZ1 = dA1 * self.A1 * (1 - self.A1)
        else:
            raise ValueError("Unsupported activation function")
        dW1 = np.dot(self.X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        # Store gradients for visualization
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features with fixed axes
    hidden_features = mlp.A1  # shape (n_samples, hidden_dim)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Space at Step {frame * 10}')
    ax_hidden.set_xlabel('Neuron 1 Activation')
    ax_hidden.set_ylabel('Neuron 2 Activation')
    ax_hidden.set_zlabel('Neuron 3 Activation')
    ax_hidden.set_xlim([-1.5, 1.5])
    ax_hidden.set_ylim([-1.5, 1.5])
    ax_hidden.set_zlim([-1.5, 1.5])

    # Hyperplane visualization in the hidden space
    # Plot hyperplanes where activations are zero (planes x=0, y=0, z=0)
    x_lim = y_lim = z_lim = np.linspace(-1.5, 1.5, 10)
    X0, Y0 = np.meshgrid(x_lim, y_lim)
    Z0 = np.zeros_like(X0)
    

    # Distorted input space transformed by the hidden layer as a plane
    # Generate grid in input space
    grid_x, grid_y = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 20),
                                 np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 20))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    # Compute hidden activations for the grid points
    Z1_grid = np.dot(grid, mlp.W1) + mlp.b1
    if mlp.activation_fn == 'tanh':
        A1_grid = np.tanh(Z1_grid)
    elif mlp.activation_fn == 'relu':
        A1_grid = np.maximum(0, Z1_grid)
    elif mlp.activation_fn == 'sigmoid':
        A1_grid = 1 / (1 + np.exp(-Z1_grid))
    else:
        raise ValueError("Unsupported activation function")
    # Reshape A1_grid to match the grid shape
    A1_grid_x = A1_grid[:, 0].reshape(grid_x.shape)
    A1_grid_y = A1_grid[:, 1].reshape(grid_x.shape)
    A1_grid_z = A1_grid[:, 2].reshape(grid_x.shape)
    # Plot the plane in the hidden space
    ax_hidden.plot_surface(A1_grid_x, A1_grid_y, A1_grid_z, color='purple', alpha=0.3)

    # Plot decision boundary in hidden layer space (output neuron decision boundary)
    # The decision boundary is where Z2 = 0, i.e., A1 @ W2 + b2 = 0
    # Since we have 3D hidden space, the decision boundary is a plane
    W = mlp.W2.flatten()
    b = mlp.b2.flatten()
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    if W[2] != 0:
        zz = (-W[0] * xx - W[1] * yy - b[0]) / W[2]
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='tan')

    # Plot input layer decision boundary without gradient fill
    xx_input, yy_input = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    grid = np.c_[xx_input.ravel(), yy_input.ravel()]
    out = mlp.forward(grid)
    Z = out.reshape(xx_input.shape)
    ax_input.contour(xx_input, yy_input, Z, levels=[0.5], colors='k', linewidths=1)
    ax_input.contourf(xx_input, yy_input, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f'Input Space at Step {frame * 10}')
    ax_input.set_xlabel('x1')
    ax_input.set_ylabel('x2')

    # Draw network structure visualization
    ax_gradient.axis('off')
    ax_gradient.set_title(f'Network Structure at Step {frame * 10}')

    # Positions of neurons in each layer
    layer_sizes = [2, 3, 1]
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.2 / float(len(layer_sizes) + 1)
    # Coordinates for neurons
    neuron_coords = {}
    colors = ['b', 'g', 'r']  # Different colors for each layer
    for i, n in enumerate(layer_sizes):
        layer_x = h_spacing * (i + 1)
        layer_y = np.linspace(0.5 - (n - 1) / 2.0 * v_spacing, 0.5 + (n - 1) / 2.0 * v_spacing, n)
        neuron_coords[i] = list(zip([layer_x] * n, layer_y))

    # Draw neurons
    for i, layer in neuron_coords.items():
        for j, (x, y) in enumerate(layer):
            circle = Circle((x, y), v_spacing / 6.0, fill=True, color=colors[i], ec='k', lw=2, alpha=0.9)
            ax_gradient.add_artist(circle)
            # Label neurons
            if i == 0:
                label = f"x{j + 1}"
            elif i == 1:
                label = f"h{j + 1}"
            else:
                label = "y"
            ax_gradient.text(x, y, label, fontsize=10, ha='center', va='center', color='white', weight='bold')

    for i in range(len(layer_sizes) - 1):
        layer_i = neuron_coords[i]
        layer_j = neuron_coords[i + 1]
        # Get the weights and gradients between layer i and layer j
        if i == 0:
            grads = mlp.dW1
        elif i == 1:
            grads = mlp.dW2
        for idx_i, (x_i, y_i) in enumerate(layer_i):
            for idx_j, (x_j, y_j) in enumerate(layer_j):
                grad = grads[idx_i, idx_j]
                color = 'purple'
                if grad >= 2:
                    linewidth = 4
                elif grad >= 1:
                    linewidth = 2.5
                elif grad >= 0:
                    linewidth = 1.5
                else:
                    linewidth = 0.5
                arrow = FancyArrowPatch((x_i, y_i), (x_j, y_j), color=color, arrowstyle='-', mutation_scale=10, lw=linewidth, alpha=0.8)
                ax_gradient.add_patch(arrow)

# Main visualization function
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(18, 6))  # Adjust figure size to make it smaller
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
