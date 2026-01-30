#!/usr/bin/env python3
"""
Diffusion Policy Inference Script for SO-101 Robot Arm

This script runs a trained diffusion policy model on an SO-101 (or SO-100) robot arm.
It handles:
- Robot connection and control via lerobot
- Camera observation capture
- Diffusion policy inference
- Action execution with temporal ensemble
- Inference speed benchmarking

Requirements:
    pip install torch torchvision opencv-python numpy

Usage:
    # Benchmark mode (no robot needed) - test inference speed
    python run_diffusion_policy_so101.py --benchmark --num-runs 100
    
    # Robot operation mode
    python run_diffusion_policy_so101.py --checkpoint path/to/checkpoint --robot-port /dev/ttyUSB0
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


# =============================================================================
# Configuration
# =============================================================================

class DiffusionPolicyConfig:
    """Configuration for diffusion policy inference."""
    
    # Observation settings
    obs_horizon: int = 2          # Number of observation steps to condition on
    action_horizon: int = 8       # Number of action steps to predict
    action_dim: int = 6           # SO-101 has 6 DOF (5 arm joints + 1 gripper)
    
    # Image settings
    image_size: tuple = (96, 96)  # Input image resolution
    num_cameras: int = 1          # Number of camera views
    
    # Diffusion settings
    num_diffusion_steps: int = 100
    num_inference_steps: int = 10  # DDIM steps for faster inference
    
    # Control settings
    control_freq: int = 30        # Hz
    temporal_ensemble: bool = True
    ensemble_weights: str = "exp"  # "exp" or "uniform"


# =============================================================================
# Diffusion Policy Model
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion timestep."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConditionalUNet1D(nn.Module):
    """
    1D UNet for diffusion policy action prediction.
    Conditioned on visual observations and robot state.
    """
    
    def __init__(
        self,
        action_dim: int = 6,
        action_horizon: int = 8,
        obs_dim: int = 512,  # Visual encoder output dim
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                )
            )
        
        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        noisy_actions: torch.Tensor,  # (B, T, action_dim)
        timestep: torch.Tensor,        # (B,)
        obs_features: torch.Tensor,    # (B, obs_dim)
    ) -> torch.Tensor:
        B, T, _ = noisy_actions.shape
        
        # Embed timestep
        t_emb = self.time_mlp(timestep)  # (B, hidden_dim)
        
        # Encode observations
        obs_emb = self.obs_encoder(obs_features)  # (B, hidden_dim)
        
        # Combine conditioning
        cond = t_emb + obs_emb  # (B, hidden_dim)
        
        # Project input
        x = self.input_proj(noisy_actions)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, T)
        
        # Add conditioning
        x = x + cond[:, :, None]
        
        # Down path
        skip_connections = []
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)
        
        # Middle
        x = self.mid_block(x)
        
        # Up path
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        # Output
        x = x.transpose(1, 2)  # (B, T, hidden_dim)
        x = self.output_proj(x)  # (B, T, action_dim)
        
        return x


class VisualEncoder(nn.Module):
    """ResNet-based visual encoder for image observations."""
    
    def __init__(self, output_dim: int = 512, num_cameras: int = 1):
        super().__init__()
        
        # Simple CNN encoder (can be replaced with ResNet)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * num_cameras, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3 * num_cameras, 96, 96)
            feat_size = self.encoder(dummy).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(feat_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.fc(features)


class DiffusionPolicy(nn.Module):
    """Complete diffusion policy model."""
    
    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(
            output_dim=512,
            num_cameras=config.num_cameras,
        )
        
        # Noise prediction network
        self.noise_pred_net = ConditionalUNet1D(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            obs_dim=512 * config.obs_horizon,  # Concatenated observation features
        )
        
        # Diffusion scheduler parameters
        self.register_buffer(
            "betas",
            torch.linspace(0.0001, 0.02, config.num_diffusion_steps)
        )
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
    
    @torch.no_grad()
    def predict_action(
        self,
        obs_images: torch.Tensor,  # (B, obs_horizon, C, H, W)
    ) -> torch.Tensor:
        """Predict actions using DDIM sampling."""
        B = obs_images.shape[0]
        device = obs_images.device
        config = self.config
        
        # Encode all observation images
        obs_features = []
        for t in range(config.obs_horizon):
            feat = self.visual_encoder(obs_images[:, t])
            obs_features.append(feat)
        obs_features = torch.cat(obs_features, dim=-1)  # (B, obs_dim * obs_horizon)
        
        # Initialize noise
        noisy_actions = torch.randn(
            B, config.action_horizon, config.action_dim,
            device=device
        )
        
        # DDIM sampling
        timesteps = torch.linspace(
            config.num_diffusion_steps - 1, 0, config.num_inference_steps,
            dtype=torch.long, device=device
        )
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            
            # Predict noise
            noise_pred = self.noise_pred_net(noisy_actions, t_batch, obs_features)
            
            # DDIM update
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Predict x0
            pred_x0 = (noisy_actions - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            
            # Compute next sample
            noisy_actions = (
                torch.sqrt(alpha_prev) * pred_x0 +
                torch.sqrt(1 - alpha_prev) * noise_pred
            )
        
        return noisy_actions


# =============================================================================
# Dummy Checkpoint Generator
# =============================================================================

def create_dummy_checkpoint(config: DiffusionPolicyConfig, save_path: str = None) -> dict:
    """
    Create a dummy checkpoint with random weights for testing.
    
    Args:
        config: Model configuration
        save_path: Optional path to save the checkpoint
        
    Returns:
        Dictionary containing model state dict
    """
    print("Creating dummy checkpoint with random weights...")
    
    # Create model
    model = DiffusionPolicy(config)
    
    # Initialize with random weights (already done by default)
    # But we can add some structure to make it more realistic
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Create checkpoint dict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "obs_horizon": config.obs_horizon,
            "action_horizon": config.action_horizon,
            "action_dim": config.action_dim,
            "num_diffusion_steps": config.num_diffusion_steps,
            "num_inference_steps": config.num_inference_steps,
        },
        "metadata": {
            "type": "dummy_checkpoint",
            "created_for": "inference_speed_testing",
        }
    }
    
    # Save if path provided
    if save_path:
        torch.save(checkpoint, save_path)
        print(f"Saved dummy checkpoint to {save_path}")
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return checkpoint


# =============================================================================
# Robot Interface
# =============================================================================

class SO101Robot:
    """Interface for SO-101 robot arm using lerobot."""
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        camera_index: int = 0,
        mock: bool = False,
    ):
        self.port = port
        self.camera_index = camera_index
        self.mock = mock
        self.robot = None
        self.camera = None
        
    def connect(self):
        """Connect to robot and camera."""
        if self.mock:
            print("Using mock robot interface (no hardware)")
            return
            
        try:
            # Try to import lerobot
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
            from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
            
            # Configure leader and follower arms
            # SO-101 typically uses Feetech STS3215 servos
            follower_config = {
                "port": self.port,
                "motors": {
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            }
            
            self.robot = ManipulatorRobot(
                robot_type="so100",
                follower_arms={"main": FeetechMotorsBus(**follower_config)},
            )
            self.robot.connect()
            print(f"Connected to SO-101 on {self.port}")
            
        except ImportError:
            print("Warning: lerobot not installed. Using mock robot interface.")
            self.robot = None
        except Exception as e:
            print(f"Warning: Could not connect to robot: {e}")
            print("Using mock robot interface.")
            self.robot = None
        
        # Initialize camera
        if not self.mock:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print(f"Warning: Could not open camera {self.camera_index}")
                self.camera = None
    
    def get_observation(self) -> dict:
        """Get current observation (image + joint positions)."""
        obs = {}
        
        # Get camera image
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                obs["image"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                obs["image"] = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Generate random image for testing
            obs["image"] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Get joint positions
        if self.robot is not None:
            state = self.robot.get_state()
            obs["joint_positions"] = np.array(state["follower"]["main"])
        else:
            obs["joint_positions"] = np.zeros(6, dtype=np.float32)
        
        return obs
    
    def send_action(self, action: np.ndarray):
        """Send action to robot."""
        if self.robot is not None:
            # Convert to joint positions and send
            self.robot.send_action({"main": action.tolist()})
    
    def disconnect(self):
        """Disconnect from robot and camera."""
        if self.robot is not None:
            self.robot.disconnect()
        if self.camera is not None:
            self.camera.release()


# =============================================================================
# Inference Speed Benchmark
# =============================================================================

class InferenceBenchmark:
    """Benchmark inference speed of diffusion policy."""
    
    def __init__(
        self,
        model: DiffusionPolicy,
        config: DiffusionPolicyConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def generate_dummy_observation(self) -> torch.Tensor:
        """Generate a dummy observation tensor."""
        # Generate random images
        images = []
        for _ in range(self.config.obs_horizon):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_tensor = self.image_transform(img)
            images.append(img_tensor)
        
        # Stack and add batch dimension
        obs = torch.stack(images, dim=0).unsqueeze(0)  # (1, obs_horizon, C, H, W)
        return obs.to(self.device)
    
    def warmup(self, num_warmup: int = 10):
        """Warmup the model to ensure accurate timing."""
        print(f"Warming up with {num_warmup} iterations...")
        obs = self.generate_dummy_observation()
        
        for _ in range(num_warmup):
            _ = self.model.predict_action(obs)
        
        # Sync CUDA if using GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def run_benchmark(
        self,
        num_runs: int = 100,
        include_preprocessing: bool = True,
    ) -> dict:
        """
        Run inference benchmark.
        
        Args:
            num_runs: Number of inference runs
            include_preprocessing: Whether to include image preprocessing in timing
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Running Inference Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Number of runs: {num_runs}")
        print(f"Include preprocessing: {include_preprocessing}")
        print(f"Observation horizon: {self.config.obs_horizon}")
        print(f"Action horizon: {self.config.action_horizon}")
        print(f"DDIM inference steps: {self.config.num_inference_steps}")
        print(f"{'='*60}\n")
        
        # Warmup
        self.warmup()
        
        # Storage for timing
        inference_times = []
        preprocess_times = []
        total_times = []
        
        # Pre-generate observation for non-preprocessing benchmark
        if not include_preprocessing:
            fixed_obs = self.generate_dummy_observation()
        
        print(f"Running {num_runs} inference iterations...")
        
        for i in range(num_runs):
            # Total timing start
            total_start = time.perf_counter()
            
            if include_preprocessing:
                # Time preprocessing
                preprocess_start = time.perf_counter()
                obs = self.generate_dummy_observation()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                preprocess_end = time.perf_counter()
                preprocess_times.append(preprocess_end - preprocess_start)
            else:
                obs = fixed_obs
            
            # Time inference
            inference_start = time.perf_counter()
            _ = self.model.predict_action(obs)
            if self.device == "cuda":
                torch.cuda.synchronize()
            inference_end = time.perf_counter()
            
            inference_times.append(inference_end - inference_start)
            total_times.append(inference_end - total_start)
            
            # Progress
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} iterations")
        
        # Calculate statistics
        inference_times = np.array(inference_times) * 1000  # Convert to ms
        total_times = np.array(total_times) * 1000
        
        results = {
            "num_runs": num_runs,
            "device": self.device,
            "inference": {
                "mean_ms": np.mean(inference_times),
                "std_ms": np.std(inference_times),
                "min_ms": np.min(inference_times),
                "max_ms": np.max(inference_times),
                "median_ms": np.median(inference_times),
                "p95_ms": np.percentile(inference_times, 95),
                "p99_ms": np.percentile(inference_times, 99),
                "hz": 1000 / np.mean(inference_times),
            },
            "total": {
                "mean_ms": np.mean(total_times),
                "std_ms": np.std(total_times),
                "hz": 1000 / np.mean(total_times),
            },
        }
        
        if include_preprocessing:
            preprocess_times = np.array(preprocess_times) * 1000
            results["preprocessing"] = {
                "mean_ms": np.mean(preprocess_times),
                "std_ms": np.std(preprocess_times),
            }
        
        # Print results
        self._print_results(results, include_preprocessing)
        
        return results
    
    def _print_results(self, results: dict, include_preprocessing: bool):
        """Print benchmark results in a formatted way."""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        inf = results["inference"]
        print(f"\n📊 Inference Time (model forward pass only):")
        print(f"   Mean:   {inf['mean_ms']:8.3f} ms")
        print(f"   Std:    {inf['std_ms']:8.3f} ms")
        print(f"   Min:    {inf['min_ms']:8.3f} ms")
        print(f"   Max:    {inf['max_ms']:8.3f} ms")
        print(f"   Median: {inf['median_ms']:8.3f} ms")
        print(f"   P95:    {inf['p95_ms']:8.3f} ms")
        print(f"   P99:    {inf['p99_ms']:8.3f} ms")
        print(f"\n   ⚡ Inference Speed: {inf['hz']:.1f} Hz")
        
        if include_preprocessing:
            pre = results["preprocessing"]
            print(f"\n📊 Preprocessing Time:")
            print(f"   Mean:   {pre['mean_ms']:8.3f} ms")
            print(f"   Std:    {pre['std_ms']:8.3f} ms")
        
        tot = results["total"]
        print(f"\n📊 Total Time (preprocessing + inference):")
        print(f"   Mean:   {tot['mean_ms']:8.3f} ms")
        print(f"   Std:    {tot['std_ms']:8.3f} ms")
        print(f"\n   ⚡ Total Speed: {tot['hz']:.1f} Hz")
        
        # Feasibility check
        print(f"\n{'='*60}")
        print("REAL-TIME FEASIBILITY")
        print(f"{'='*60}")
        
        target_freqs = [10, 20, 30, 50]
        for freq in target_freqs:
            required_ms = 1000 / freq
            achievable = inf['mean_ms'] < required_ms
            status = "✅" if achievable else "❌"
            margin = required_ms - inf['mean_ms']
            print(f"   {freq:2d} Hz ({required_ms:5.1f} ms budget): {status} (margin: {margin:+.1f} ms)")
        
        print(f"{'='*60}\n")


# =============================================================================
# Inference Loop
# =============================================================================

class DiffusionPolicyRunner:
    """Runs diffusion policy inference on SO-101."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: DiffusionPolicyConfig,
        robot: SO101Robot,
        device: str = "cuda",
    ):
        self.config = config
        self.robot = robot
        self.device = device
        
        # Load model
        self.model = DiffusionPolicy(config).to(device)
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Warning: No checkpoint loaded. Using random weights.")
        self.model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Observation buffer
        self.obs_buffer = deque(maxlen=config.obs_horizon)
        
        # Action queue for temporal ensemble
        self.action_queue = deque(maxlen=config.action_horizon)
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        return self.image_transform(image)
    
    def get_ensemble_action(self, new_actions: np.ndarray) -> np.ndarray:
        """Apply temporal ensemble to smooth actions."""
        config = self.config
        
        if not config.temporal_ensemble:
            return new_actions[0]
        
        # Add new predictions to queue
        self.action_queue.append(new_actions)
        
        if len(self.action_queue) == 1:
            return new_actions[0]
        
        # Compute weighted average
        if config.ensemble_weights == "exp":
            # Exponential weighting (more recent = higher weight)
            weights = np.exp(np.arange(len(self.action_queue)))
        else:
            weights = np.ones(len(self.action_queue))
        weights = weights / weights.sum()
        
        # Ensemble across predictions
        ensemble_action = np.zeros(config.action_dim)
        for i, (w, actions) in enumerate(zip(weights, self.action_queue)):
            # Each prediction in queue predicts the next action_horizon steps
            # We want the action for the current timestep
            action_idx = len(self.action_queue) - 1 - i
            if action_idx < len(actions):
                ensemble_action += w * actions[action_idx]
        
        return ensemble_action
    
    @torch.no_grad()
    def step(self) -> np.ndarray:
        """Execute one control step."""
        # Get observation
        obs = self.robot.get_observation()
        image_tensor = self.preprocess_image(obs["image"])
        
        # Add to buffer
        self.obs_buffer.append(image_tensor)
        
        # Pad buffer if needed
        while len(self.obs_buffer) < self.config.obs_horizon:
            self.obs_buffer.appendleft(self.obs_buffer[0])
        
        # Stack observations
        obs_images = torch.stack(list(self.obs_buffer), dim=0)  # (obs_horizon, C, H, W)
        obs_images = obs_images.unsqueeze(0).to(self.device)     # (1, obs_horizon, C, H, W)
        
        # Predict actions
        actions = self.model.predict_action(obs_images)
        actions = actions[0].cpu().numpy()  # (action_horizon, action_dim)
        
        # Apply temporal ensemble
        action = self.get_ensemble_action(actions)
        
        return action
    
    def run(self, num_steps: int = None, duration: float = None):
        """
        Run inference loop.
        
        Args:
            num_steps: Number of steps to run (if specified)
            duration: Duration in seconds to run (if specified)
        """
        config = self.config
        dt = 1.0 / config.control_freq
        
        print(f"Starting inference at {config.control_freq} Hz...")
        print("Press Ctrl+C to stop")
        
        step_count = 0
        start_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # Check termination conditions
                if num_steps is not None and step_count >= num_steps:
                    break
                if duration is not None and (time.time() - start_time) >= duration:
                    break
                
                # Execute step
                action = self.step()
                self.robot.send_action(action)
                
                step_count += 1
                
                # Maintain control frequency
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                
                # Print status periodically
                if step_count % config.control_freq == 0:
                    actual_freq = step_count / (time.time() - start_time)
                    print(f"Step {step_count}, Actual freq: {actual_freq:.1f} Hz")
                    
        except KeyboardInterrupt:
            print("\nStopping inference...")
        
        elapsed = time.time() - start_time
        print(f"Completed {step_count} steps in {elapsed:.1f}s ({step_count/elapsed:.1f} Hz)")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run diffusion policy on SO-101 robot arm"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for robot connection"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for observation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=30,
        help="Control frequency in Hz"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to run in seconds"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of steps to run"
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable temporal ensemble"
    )
    # Benchmark mode arguments
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference speed benchmark (no robot needed)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs for benchmark"
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Path to save dummy checkpoint"
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=10,
        help="Number of DDIM inference steps"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup config
    config = DiffusionPolicyConfig()
    config.control_freq = args.control_freq
    config.temporal_ensemble = not args.no_ensemble
    config.num_inference_steps = args.ddim_steps
    
    # Create dummy checkpoint if needed
    if args.save_checkpoint or (args.benchmark and args.checkpoint is None):
        checkpoint = create_dummy_checkpoint(config, args.save_checkpoint)
    
    # Benchmark mode
    if args.benchmark:
        print("\n🚀 Running in BENCHMARK mode\n")
        
        # Create model
        model = DiffusionPolicy(config).to(args.device)
        
        # Load checkpoint if provided, otherwise use random weights
        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location=args.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print("Using random weights for benchmark")
        
        model.eval()
        
        # Run benchmark
        benchmark = InferenceBenchmark(model, config, args.device)
        results = benchmark.run_benchmark(
            num_runs=args.num_runs,
            include_preprocessing=True,
        )
        
        return
    
    # Normal robot operation mode
    robot = SO101Robot(
        port=args.robot_port,
        camera_index=args.camera,
    )
    robot.connect()
    
    try:
        # Initialize runner
        runner = DiffusionPolicyRunner(
            checkpoint_path=args.checkpoint,
            config=config,
            robot=robot,
            device=args.device,
        )
        
        # Run inference
        runner.run(
            num_steps=args.num_steps,
            duration=args.duration,
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()