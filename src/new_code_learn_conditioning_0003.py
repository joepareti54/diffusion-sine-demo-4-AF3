"""
Conditional Diffusion Model for Sine Wave Generation

A minimal implementation of a conditional diffusion model for generating sine waves.
Uses pure input data approach similar to AF3, with improved training stability.

The program implements:
1. A neural network (ConditionedDenoiseModel) that learns to denoise data
2. A diffusion process (Diffusion) that handles noise addition and removal
3. Training data generation and model training utilities
4. Visualization of results

Key Features:
- 1000-step diffusion process
- Enhanced neural network architecture (101→256→256→256→100)
- Improved training stability with gradient clipping
- Multiple training steps per batch
- Pure input data approach
- Binary conditioning (-1 or 1) for wave orientation

Dependencies:
    torch: Neural network and tensor operations
    torch.nn: Neural network modules
    matplotlib.pyplot: Visualization
    numpy: Numerical operations

Main Components:

1. ConditionedDenoiseModel:
   - Input: 101 dimensions (100 points + 1 condition)
   - Hidden layers: 3x256 units with ReLU
   - Output: 100 dimensions
   - Conditioning: Concatenation-based

2. Diffusion:
   - 1000 diffusion steps
   - Linear noise schedule (beta: 1e-5 to 0.01)
   - Forward process: add_noise()
   - Reverse process: sample()

3. Training:
   - 2000 epochs
   - Batch size: 64
   - Learning rate: 1e-5
   - Adam optimizer
   - MSE loss
   - Gradient clipping at 1.0
   - 4 training steps per batch

4. Data Generation:
   - 2000 samples
   - Pure sine waves
   - Binary conditions (-1, 1)
   - Target: regular/inverted waves

5. Visualization:
   - Three subplots:
     a. Input sine wave
     b. Generated waves (same noise, different conditions)
     c. Target waves
   - Grid and legend for clarity
   - Fixed y-axis limits (-1.2 to 1.2)

Usage:
    Run the script directly to train the model and visualize results:
    $ python new_code_learn_conditioning_0003.py

Output:
    - Training progress (loss every 100 epochs)
    - Three-panel visualization of results

Author: [Your Name]
Version: 0.0.3
"""
#
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class ConditionedDenoiseModel(nn.Module):
    """
    Neural network model that learns to denoise data conditioned on a binary input.
    Enhanced architecture with increased capacity for better learning.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(101, 256),  # 100 points + 1 condition
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 100)   # Output: 100 points
        )
    
    def forward(self, x, condition):
        x_input = torch.cat([x, condition], dim=1)
        return self.net(x_input)

class Diffusion:
    """
    Implements the diffusion process for adding and removing noise from data.
    Modified for more stable diffusion process.
    """
    def __init__(self, n_steps=1000):
        self.n_steps = n_steps
        self.betas = torch.linspace(1e-5, 0.01, n_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return noisy_x, noise

    def sample(self, model, condition, n_steps=None):
        if n_steps is None:
            n_steps = self.n_steps
            
        x = torch.randn(1, 100)  # Start from noise
        
        for t in reversed(range(n_steps)):
            pred_noise = model(x, condition.view(1, 1))
            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod) * pred_noise
            ) + torch.sqrt(self.betas[t]) * noise
            
        return x

def generate_training_data(n_samples=2000):
    """
    Generate training data using pure input approach.
    Returns separate input data, conditions, and targets.
    """
    x = np.linspace(0, 10, 100)
    base_sine = torch.FloatTensor(np.sin(x))
    
    # Input data: always regular sine waves
    input_data = base_sine.repeat(n_samples, 1)
    
    # Conditions: -1 or 1
    conditions = (torch.randint(0, 2, (n_samples, 1)) * 2 - 1).float()
    
    # Target data: regular or inverted sine waves based on conditions
    
    target_data = base_sine.repeat(n_samples, 1) * conditions
    return input_data, conditions, target_data

def train_model(n_epochs=2000, batch_size=64):
    """
    Train the diffusion model with improved stability measures.
    """
    model = ConditionedDenoiseModel()
    diffusion = Diffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    input_data, conditions, target_data = generate_training_data()
    
    for epoch in range(n_epochs):
        total_loss = 0
        batches = 0
        
        for i in range(0, len(input_data), batch_size):
            input_batch = input_data[i:i+batch_size]
            batch_conditions = conditions[i:i+batch_size]
            target_batch = target_data[i:i+batch_size]
            
            # Multiple training steps per batch
            for _ in range(4):
                t = torch.randint(0, diffusion.n_steps, (input_batch.shape[0],))
                noisy_batch, noise = diffusion.add_noise(target_batch, t)
                pred_noise = model(noisy_batch, batch_conditions)
                
                loss = nn.MSELoss()(pred_noise, noise)
                total_loss += loss.item()
                batches += 1
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Added gradient clipping
                optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / batches
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}')
    
    return model, diffusion

def visualize_results(model, diffusion):
    """
    Visualize the results of the diffusion model.
    """
    model.eval()
    with torch.no_grad():
        plt.figure(figsize=(15, 5))
        
        # Show the input sine wave
        plt.subplot(131)
        x = np.linspace(0, 10, 100)
        plt.plot(np.sin(x))
        plt.title('Input Sine Wave')
        plt.grid(True)
        plt.ylim(-1.2, 1.2)
        
        # Generate with both conditions from the SAME noise
        plt.subplot(132)
        condition_regular = torch.tensor([1.0])
        condition_inverted = torch.tensor([-1.0])
        
        # Use same noise for both generations
        torch.manual_seed(42)  # For reproducibility
        noise = torch.randn(1, 100)
        
        torch.manual_seed(42)  # Reset seed to use same noise
        generated_regular = diffusion.sample(model, condition_regular)
        torch.manual_seed(42)  # Reset seed to use same noise
        generated_inverted = diffusion.sample(model, condition_inverted)
        
        plt.plot(generated_regular[0].numpy(), label='Regular (condition=1)')
        plt.plot(generated_inverted[0].numpy(), label='Inverted (condition=-1)')
        plt.title('Generated Waves\nfrom Same Noise')
        plt.legend()
        plt.grid(True)
        plt.ylim(-1.2, 1.2)
        
        # Show target waves
        plt.subplot(133)
        plt.plot(np.sin(x), label='Target Regular')
        plt.plot(-np.sin(x), label='Target Inverted')
        plt.title('Target Waves')
        plt.legend()
        plt.grid(True)
        plt.ylim(-1.2, 1.2)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to train the model and visualize results.
    """
    print("Training model...")
    model, diffusion = train_model()
    
    print("\nGenerating visualizations...")
    visualize_results(model, diffusion)

if __name__ == "__main__":
    main()
