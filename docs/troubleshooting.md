# Troubleshooting

Common issues and solutions for the Minecraft 1.8.9 GPU simulator.

## Installation Issues

### "VK_ICD_FILENAMES not set"

**Symptom:** Vulkan initialization fails with `vkCreateInstance` returning `VK_ERROR_INCOMPATIBLE_DRIVER` or "Failed to create Vulkan instance".

**Fix:**

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

Add to your shell profile for persistence:

```bash
echo 'export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json' >> ~/.zshrc
```

**Diagnostics:**

```bash
# Verify Vulkan is accessible
vulkaninfo --summary

# Check MoltenVK installation
ls /opt/homebrew/etc/vulkan/icd.d/

# Verify the ICD file points to a valid library
cat $VK_ICD_FILENAMES
```

### "mc189_core not found"

**Symptom:** `ImportError: No module named 'mc189_core'` when importing the simulator.

**Fix:** Build the C++ extension and copy the `.so` file to the Python package directory:

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3.12)
make -j$(nproc)
cp mc189_core*.so ../../python/minecraft_sim/
```

**Diagnostics:**

```bash
# Check if the .so file exists
find . -name "mc189_core*"

# Check library dependencies
otool -L python/minecraft_sim/mc189_core*.so  # macOS
ldd python/minecraft_sim/mc189_core*.so       # Linux

# Verify Python can import it
uv run python -c "import mc189_core; print(mc189_core.__file__)"
```

### "Python version mismatch"

**Symptom:** `ImportError` with ABI mismatch errors like `undefined symbol: _PyObject_GC_IS_TRACKED` or references to `cpython-314` when the extension targets `cpython-312`.

**Fix:** Use Python 3.12 exclusively. The C++ extensions are compiled against the Python 3.12 ABI only. Do not use Python 3.14 (system default on macOS).

```bash
# Use Python 3.12 explicitly
uv venv --python 3.12
source .venv/bin/activate

# Rebuild the extension with the correct Python
cd cpp/build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make clean && make -j$(nproc)
```

**Diagnostics:**

```bash
# Check which Python is active
python --version

# Check the extension's target Python
file python/minecraft_sim/mc189_core*.so

# Verify ABI compatibility
uv run python -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))"
```

## Runtime Issues

### "Low throughput"

**Symptom:** Steps per second significantly below expected (~10k+ for GPU backend, < 1000 indicates a problem).

**Diagnostics:**

```bash
# Check GPU utilization
nvidia-smi dmon -s u -d 1  # Linux/NVIDIA

# macOS Metal GPU usage
sudo powermetrics --samplers gpu_power -i 1000 -n 5

# Profile shader execution time
uv run python -c "
from minecraft_sim import SpeedrunVecEnv
env = SpeedrunVecEnv(num_envs=64)
import time
start = time.perf_counter()
for _ in range(1000):
    env.step(env.action_space.sample())
elapsed = time.perf_counter() - start
print(f'{64000/elapsed:.0f} steps/sec')
"
```

**Fixes:**

- Reduce `num_envs` if GPU memory is saturated (watch for allocation failures in logs)
- Verify shaders are pre-compiled (check for `.spv` files in `cpp/shaders/`)
- Ensure no CPU fallback is active; check logs for "falling back to CPU"
- On macOS, close other Metal-heavy applications
- Use `VK_KHR_timeline_semaphore` for better pipeline overlap

### "NaN in observations"

**Symptom:** Observation tensors contain NaN or Inf values, causing training instability.

**Diagnostics:**

```bash
uv run python -c "
import numpy as np
from minecraft_sim import SpeedrunVecEnv
env = SpeedrunVecEnv(num_envs=4)
obs = env.reset()
for i in range(100):
    obs, rew, done, info = env.step(env.action_space.sample())
    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        print(f'NaN/Inf at step {i}')
        print(f'  NaN indices: {np.argwhere(np.isnan(obs))}')
        print(f'  Reward: {rew}')
        print(f'  Done: {done}')
        break
"
```

**Fixes:**

- Check for division by zero in reward shaping (particularly distance-based rewards when at target)
- Validate that `env.reset()` produces finite observations after episode termination
- Add `np.clip` to observation normalization if values can grow unbounded
- Check the observation encoder for uninitialized memory in the GPU buffer
- Enable `VK_LAYER_KHRONOS_validation` to catch shader output issues

### "Agent stuck in death loop"

**Symptom:** Agent repeatedly dies and respawns without making progress. Episode rewards are consistently negative.

**Diagnostics:**

```bash
uv run python -c "
from minecraft_sim import SpeedrunVecEnv
env = SpeedrunVecEnv(num_envs=1)
obs = env.reset()
deaths = 0
for i in range(1000):
    obs, rew, done, info = env.step(env.action_space.sample())
    if done[0]:
        deaths += 1
        if info[0].get('death', False):
            print(f'Death #{deaths} at step {i}: cause={info[0].get(\"death_cause\", \"unknown\")}')
print(f'Total deaths in 1000 steps: {deaths}')
"
```

**Fixes:**

- Clip death penalty: use `-1.0` not `-100.0`; large penalties cause the agent to learn "don't exist"
- Add a survival bonus: `+0.01` per tick alive to counterbalance death penalty
- Add spawn protection: skip damage for the first 20 ticks after respawn
- Check if spawn location is inside blocks or above a void

## Training Issues

### "Reward collapse to zero"

**Symptom:** All environments report zero or near-zero reward after initial training period.

**Diagnostics:**

```bash
# Check reward statistics during training
uv run python -c "
import numpy as np
from minecraft_sim import SpeedrunVecEnv
env = SpeedrunVecEnv(num_envs=32)
obs = env.reset()
rewards = []
for i in range(500):
    obs, rew, done, info = env.step(env.action_space.sample())
    rewards.append(rew)
rewards = np.array(rewards)
print(f'Mean: {rewards.mean():.6f}')
print(f'Std:  {rewards.std():.6f}')
print(f'Max:  {rewards.max():.6f}')
print(f'Min:  {rewards.min():.6f}')
print(f'Nonzero fraction: {(rewards != 0).mean():.4f}')
"
```

**Fixes:**

- Lower the learning rate (try 1e-4 or 3e-5)
- Increase entropy coefficient to encourage exploration (0.01 -> 0.05)
- Add intrinsic curiosity or count-based exploration bonuses
- Verify reward signals fire at all with random actions (see diagnostic above)
- Check for reward normalization squashing sparse signals

### "Episode length not decreasing"

**Symptom:** Agent completes episodes but average episode length plateaus, indicating no speedrun improvement.

**Diagnostics:**

```bash
uv run python -c "
from minecraft_sim import SpeedrunVecEnv
env = SpeedrunVecEnv(num_envs=8)
obs = env.reset()
episode_lengths = []
current_lengths = [0] * 8
for i in range(5000):
    obs, rew, done, info = env.step(env.action_space.sample())
    for j in range(8):
        current_lengths[j] += 1
        if done[j]:
            episode_lengths.append(current_lengths[j])
            current_lengths[j] = 0
if episode_lengths:
    import numpy as np
    lens = np.array(episode_lengths)
    print(f'Episodes: {len(lens)}')
    print(f'Mean length: {lens.mean():.0f}')
    print(f'Min: {lens.min()}, Max: {lens.max()}')
    print(f'Std: {lens.std():.0f}')
"
```

**Fixes:**

- Verify the success/completion condition triggers correctly
- Add milestone rewards for intermediate progress (crafting table, furnace, iron pickaxe, etc.)
- Add a small time penalty (`-0.001` per step) to incentivize faster completion
- Check that `max_episode_steps` isn't set too low, causing timeout before possible success
- Ensure curriculum stage transitions when metrics plateau

### "Gradient explosion"

**Symptom:** Loss becomes NaN or extremely large values, training diverges.

**Diagnostics:**

```bash
# Check for extreme observation/reward values that could cause gradient issues
uv run python -c "
import numpy as np
from minecraft_sim import SpeedrunVecEnv
env = SpeedrunVecEnv(num_envs=16)
obs = env.reset()
print(f'Obs range: [{obs.min():.2f}, {obs.max():.2f}]')
print(f'Obs shape: {obs.shape}, dtype: {obs.dtype}')
for i in range(200):
    obs, rew, done, info = env.step(env.action_space.sample())
    if abs(rew).max() > 100:
        print(f'Large reward at step {i}: {rew.max():.2f}')
    if abs(obs).max() > 1000:
        print(f'Large obs at step {i}: {obs.max():.2f}')
"
```

**Fixes:**
- Enable gradient clipping: `max_grad_norm=0.5`
- Normalize observations to [-1, 1] range
- Clip rewards to [-10, 10]
- Reduce learning rate
- Check for reward spikes on episode boundaries

## Shader Issues

### "Shader compilation failed"

**Symptom:** `vkCreateShaderModule` fails or `glslangValidator` reports errors.

**Diagnostics:**

```bash
# Validate all shaders
for shader in cpp/shaders/*.comp; do
    echo "Checking $shader..."
    glslangValidator -V "$shader" -o /dev/null 2>&1 | head -5
done

# Check for missing SPIR-V files
ls cpp/shaders/*.spv

# Recompile all shaders
cd cpp
bash compile_shaders.sh
```

### "Validation layer errors"

**Symptom:** Vulkan validation layers report errors during execution.

**Diagnostics:**

```bash
# Enable validation layers
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT

# Run with validation
uv run python -c "from minecraft_sim import SpeedrunVecEnv; env = SpeedrunVecEnv(num_envs=1); env.reset()"
```

**Fixes:**
- Check descriptor set bindings match shader expectations
- Verify buffer sizes match shader memory layout
- Ensure proper synchronization between compute dispatches
