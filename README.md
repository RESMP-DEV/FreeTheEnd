# FreeTheEnd

```
    ███████╗██████╗ ███████╗███████╗    ████████╗██╗  ██╗███████╗
    ██╔════╝██╔══██╗██╔════╝██╔════╝    ╚══██╔══╝██║  ██║██╔════╝
    █████╗  ██████╔╝█████╗  █████╗         ██║   ███████║█████╗  
    ██╔══╝  ██╔══██╗██╔══╝  ██╔══╝         ██║   ██╔══██║██╔══╝  
    ██║     ██║  ██║███████╗███████╗       ██║   ██║  ██║███████╗
    ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚══════╝
                    ███████╗███╗   ██╗██████╗ 
                    ██╔════╝████╗  ██║██╔══██╗
                    █████╗  ██╔██╗ ██║██║  ██║
                    ██╔══╝  ██║╚██╗██║██║  ██║
                    ███████╗██║ ╚████║██████╔╝
                    ╚══════╝╚═╝  ╚═══╝╚═════╝ 
```

GPU-accelerated Minecraft 1.8.9 simulator for RL speedrun training.

## Requirements

Almost any modern computer works. No datacenter hardware needed.

- Apple M1+ (integrated GPU)
- NVIDIA GTX 1060+
- AMD RX 580+
- Intel UHD 630+ (experimental)

If your computer was made after 2017 and has Vulkan 1.2, it probably runs.

## Why This Exists

Minecraft runs at 20 ticks per second. For RL training, you'd need months to train an agent through a full speedrun.

The bottleneck isn't game logic—it's rendering. Minecraft spends most of its time drawing pixels that an RL agent never sees. You don't need to draw anything to simulate physics.

FreeTheEnd strips out all rendering. Just physics: block interactions, entity movement, inventory, combat. Reimplemented as Vulkan compute shaders running hundreds of environments in parallel.

## Throughput

| Environments | Steps/sec |
|--------------|-----------|
| 64 | 264,000 |
| 256 | 615,000 |

## Install

```bash
git clone https://github.com/RESMP-DEV/FreeTheEnd.git
cd FreeTheEnd

uv venv --python 3.12 && source .venv/bin/activate
uv pip install numpy gymnasium stable-baselines3

cd cpp/build && cmake .. && cmake --build . --target mc189_core && cd ../..
cp cpp/build/mc189_core.cpython-*-darwin.so python/minecraft_sim/

# macOS
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

### CPU Backend

For CI pipelines or development without GPU:

```bash
# Build CPU-only (no Vulkan required)
cd cpp/build
cmake -DCPU_ONLY=ON ..
cmake --build . --target mc189_core
```

Python usage:
```python
import mc189_core
cfg = mc189_core.SimulatorConfig()
cfg.use_cpu = True
sim = mc189_core.MC189Simulator(cfg)
```

## Usage

```python
from minecraft_sim import SB3VecDragonFightEnv
from stable_baselines3 import PPO

env = SB3VecDragonFightEnv(num_envs=64)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

## Curriculum

| Stage | Objective |
|-------|-----------|
| 1 | Basic survival, tool crafting |
| 2 | Iron mining, armor |
| 3 | Nether portal, fortress, blaze rods |
| 4 | Enderman hunting, ender pearls |
| 5 | Stronghold location, portal activation |
| 6 | Ender Dragon fight |

## Documentation

- [Training Guide](docs/training_guide.md)
- [API Reference](docs/api.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Project Status](STATUS.md)
- [Full docs](docs/index.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
