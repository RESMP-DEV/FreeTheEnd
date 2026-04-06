#!/usr/bin/env python3
"""Add Minecraft simulation tasks to AlphaHENG queue."""

from pathlib import Path

import yaml
from alphaheng.coordinator.task_queue import Priority, Task, TaskQueue

q = TaskQueue(redis_url="redis://localhost:6379")
tasks_dir = Path("contrib/minecraft_sim/tasks")

# Task files to load
task_files = [
    "cpp_backend_full.yaml",  # C++ backend tasks
]

priority_map = {
    "P0": Priority.P0,
    "P1": Priority.P1,
    "P2": Priority.P2,
    "P3": Priority.P3,
}

for filename in task_files:
    filepath = tasks_dir / filename
    if not filepath.exists():
        print(f"Skipping (not found): {filename}")
        continue

    with open(filepath) as f:
        data = yaml.safe_load(f)

    tasks = data.get("tasks", [])
    print(f"\nLoading {len(tasks)} tasks from {filename}:")

    for task_dict in tasks:
        task = Task(
            name=task_dict["name"],
            prompt=task_dict["prompt"],
            _priority=priority_map.get(task_dict.get("priority", "P1"), Priority.P1),
            dependencies=task_dict.get("dependencies", []),
        )
        q.add_task(task)
        print(f"  Added: {task.name}")

stats = q.stats()
print("\nQueue Status:")
for key, value in stats.items():
    print(f"  {key}: {value}")


stats = q.stats()
print("\nQueue Status:")
for key, value in stats.items():
    print(f"  {key}: {value}")
