
# AIME SERGE TUYISHIME
# Search Engines and Data Mining Lecture & Lab
# CST 345 | September 6, 2025
# Perceptron Algorithm for Optimal Furniture Placement

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple
class Room:
    """Represents a room with doors, windows, and existing furniture."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.doors = []      # List of (x, y, width, height) tuples
        self.windows = []    # List of (x, y, width, height) tuples  
        self.furniture = []  # List of (x, y, width, height) tuples
        
    def add_door(self, x: float, y: float, width: float, height: float):
        self.doors.append((x, y, width, height))
        
    def add_window(self, x: float, y: float, width: float, height: float):
        self.windows.append((x, y, width, height))
        
    def add_furniture(self, x: float, y: float, width: float, height: float):
        self.furniture.append((x, y, width, height))

def distance_to_nearest(x: float, y: float, objects: List[Tuple]) -> float:
    """Calculate minimum distance from point to any object."""
    if not objects:
        return float('inf')
    
    min_dist = float('inf')
    for obj_x, obj_y, obj_w, obj_h in objects:
        center_x = obj_x + obj_w / 2
        center_y = obj_y + obj_h / 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        min_dist = min(min_dist, dist)
    
    return min_dist

def extract_features(x: float, y: float, room: Room) -> np.ndarray:
    """Extract normalized features for a position in the room."""
    
    max_dist = np.sqrt(room.width**2 + room.height**2)
    
    # x1: Distance from nearest wall
    wall_dist = min(x, y, room.width - x, room.height - y)
    x1 = wall_dist / max_dist
    
    # x2: Distance from nearest door
    door_dist = distance_to_nearest(x, y, room.doors)
    x2 = min(door_dist / max_dist, 1.0)
    
    # x3: Distance from nearest window
    window_dist = distance_to_nearest(x, y, room.windows)
    x3 = min(window_dist / max_dist, 1.0)
    
    # x4: Distance from other furniture
    furniture_dist = distance_to_nearest(x, y, room.furniture)
    x4 = min(furniture_dist / max_dist, 1.0)
    
    # x5: Centrality (distance from room center)
    center_x, center_y = room.width / 2, room.height / 2
    center_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    x5 = center_dist / max_dist
    
    return np.array([x1, x2, x3, x4, x5])

class Perceptron:
    """Simple perceptron for binary classification."""
    
    def __init__(self, n_features: int, learning_rate: float = 0.1):
        # Step 1: Randomly initialize the weights
        self.weights = np.random.uniform(-1, 1, n_features)
        self.learning_rate = learning_rate
        self.training_history = []
        
    def predict(self, x: np.ndarray) -> int:
        """Step 3: Present values x1...xn to compute output y."""
        activation = np.dot(self.weights, x)
        return 1 if activation >= 0 else -1
    
    def train_step(self, x: np.ndarray, target: int) -> bool:
        """Perform one training step. Returns True if weight update occurred."""
        prediction = self.predict(x)
        
        # Step 4: Adjust weights if prediction is wrong
        if prediction != target:
            if target == 1 and prediction == -1:  # y < 0, should be positive
                self.weights += self.learning_rate * x  # add ηxi to each wi
            elif target == -1 and prediction == 1:  # y > 0, should be negative  
                self.weights -= self.learning_rate * x  # subtract ηxi from each wi
            return True
        return False
    
    def train(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 1000) -> dict:
        """Step 5: Repeat until perceptron predicts all examples correctly."""
        n_samples = len(X)
        epoch = 0
        
        while epoch < max_epochs:
            errors = 0
            # Step 2: Select input/output pairs at random
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                if self.train_step(X[idx], y[idx]):
                    errors += 1
            
            accuracy = 1 - errors / n_samples
            self.training_history.append({
                'epoch': epoch,
                'errors': errors,
                'accuracy': accuracy,
                'weights': self.weights.copy()
            })
            
            # Check for convergence
            if errors == 0:
                print(f"Convergence achieved at epoch {epoch}")
                break
                
            epoch += 1
        
        return {
            'converged': errors == 0,
            'final_epoch': epoch,
            'final_accuracy': accuracy
        }

def is_good_placement(x: float, y: float, room: Room) -> bool:
    """Ground truth function for good furniture placement."""
    
    # Rule 1: Not too close to doors
    for door in room.doors:
        door_center_x = door[0] + door[2] / 2
        door_center_y = door[1] + door[3] / 2
        if np.sqrt((x - door_center_x)**2 + (y - door_center_y)**2) < 2.0:
            return False
    
    # Rule 2: Not too close to existing furniture
    for furn in room.furniture:
        furn_center_x = furn[0] + furn[2] / 2
        furn_center_y = furn[1] + furn[3] / 2
        if np.sqrt((x - furn_center_x)**2 + (y - furn_center_y)**2) < 1.5:
            return False
    
    # Rule 3: Not too close to walls
    wall_clearance = 0.5
    if (x < wall_clearance or y < wall_clearance or 
        x > room.width - wall_clearance or y > room.height - wall_clearance):
        return False
    
    # Rule 4: Prefer positions near windows
    near_window = False
    for window in room.windows:
        win_center_x = window[0] + window[2] / 2
        win_center_y = window[1] + window[3] / 2
        if np.sqrt((x - win_center_x)**2 + (y - win_center_y)**2) < 3.0:
            near_window = True
            break
    
    return near_window

# Setup room
room = Room(10, 8)
room.add_door(0, 3, 0.2, 2)      # Door on left wall
room.add_window(5, 8, 3, 0.2)    # Window on top wall  
room.add_window(10, 2, 0.2, 2)   # Window on right wall
room.add_furniture(1, 1, 2, 1)   # Existing sofa
room.add_furniture(8, 6, 1.5, 2) # Existing table

print("Room setup:")
print(f"Dimensions: {room.width} x {room.height}")
print(f"Doors: {len(room.doors)}, Windows: {len(room.windows)}, Furniture: {len(room.furniture)}")

# Generate training data
np.random.seed(42)
n_samples = 200
X_train = []
y_train = []

for i in range(n_samples):
    x_pos = np.random.uniform(0, room.width)
    y_pos = np.random.uniform(0, room.height)
    
    features = extract_features(x_pos, y_pos, room)
    X_train.append(features)
    
    label = 1 if is_good_placement(x_pos, y_pos, room) else -1
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\nTraining data: {n_samples} samples")
print(f"Good placements: {np.sum(y_train == 1)}")
print(f"Bad placements: {np.sum(y_train == -1)}")

# Train perceptron
perceptron = Perceptron(n_features=5, learning_rate=0.1)
print(f"\nInitial weights: {perceptron.weights}")

training_results = perceptron.train(X_train, y_train, max_epochs=100)

print(f"\nTraining Results:")
print(f"Converged: {training_results['converged']}")
print(f"Final epoch: {training_results['final_epoch']}")
print(f"Final accuracy: {training_results['final_accuracy']:.3f}")
print(f"Final weights: {perceptron.weights}")

# Analyze weights
feature_names = ['Wall Distance', 'Door Distance', 'Window Distance', 'Furniture Distance', 'Centrality']
print(f"\nWeight Analysis:")
for i, (name, weight) in enumerate(zip(feature_names, perceptron.weights)):
    print(f"w{i+1} ({name}): {weight:.3f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Perceptron Learning for Furniture Placement', fontsize=16)

# Plot 1: Room layout with training data
ax1 = axes[0, 0]
ax1.set_xlim(0, room.width)
ax1.set_ylim(0, room.height)
ax1.set_title('Room Layout with Training Data')

# Draw room elements
room_rect = Rectangle((0, 0), room.width, room.height, 
                     linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
ax1.add_patch(room_rect)

for door in room.doors:
    door_rect = Rectangle((door[0], door[1]), door[2], door[3],
                        linewidth=2, edgecolor='brown', facecolor='brown')
    ax1.add_patch(door_rect)

for window in room.windows:
    window_rect = Rectangle((window[0], window[1]), window[2], window[3],
                          linewidth=2, edgecolor='blue', facecolor='lightblue')
    ax1.add_patch(window_rect)

for furn in room.furniture:
    furn_rect = Rectangle((furn[0], furn[1]), furn[2], furn[3],
                        linewidth=2, edgecolor='red', facecolor='lightcoral')
    ax1.add_patch(furn_rect)

ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.grid(True, alpha=0.3)

# Plot 2: Decision boundary
ax2 = axes[0, 1]
xx, yy = np.meshgrid(np.linspace(0, room.width, 50),
                    np.linspace(0, room.height, 50))

grid_predictions = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        features = extract_features(xx[i, j], yy[i, j], room)
        grid_predictions[i, j] = perceptron.predict(features)

im = ax2.imshow(grid_predictions, extent=[0, room.width, 0, room.height], 
               origin='lower', cmap='RdYlGn', alpha=0.6)
ax2.set_title('Learned Decision Boundary')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
plt.colorbar(im, ax=ax2, label='Prediction (-1: Bad, +1: Good)')

# Plot 3: Training progress
ax3 = axes[1, 0]
epochs = [h['epoch'] for h in perceptron.training_history]
accuracies = [h['accuracy'] for h in perceptron.training_history]
errors = [h['errors'] for h in perceptron.training_history]

ax3_twin = ax3.twinx()
ax3.plot(epochs, accuracies, 'b-', linewidth=2, label='Accuracy')
ax3_twin.plot(epochs, errors, 'r-', linewidth=2, label='Errors')

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy', color='b')
ax3_twin.set_ylabel('Number of Errors', color='r')
ax3.set_title('Training Progress')
ax3.grid(True, alpha=0.3)

# Plot 4: Weight evolution
ax4 = axes[1, 1]
weight_history = np.array([h['weights'] for h in perceptron.training_history])

for i, name in enumerate(feature_names):
    ax4.plot(epochs, weight_history[:, i], linewidth=2, label=f'{name}')

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Weight Value')
ax4.set_title('Weight Evolution During Training')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test on new positions
test_positions = [
    (2, 6, "Near window, away from door"),
    (1, 3, "Near door"),
    (7, 4, "Center area"),
    (9, 7, "Corner area")
]

print(f"\nTest Predictions:")
for x, y, description in test_positions:
    features = extract_features(x, y, room)
    prediction = perceptron.predict(features)
    ground_truth = 1 if is_good_placement(x, y, room) else -1
    
    result = "✓" if prediction == ground_truth else "✗"
    pred_text = "Good" if prediction == 1 else "Bad"
    truth_text = "Good" if ground_truth == 1 else "Bad"
    
    print(f"({x:.1f}, {y:.1f}) - {description}: {pred_text} | Truth: {truth_text} | {result}")