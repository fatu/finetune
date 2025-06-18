"""                                                                                                                                                               │ │
AppWorld Environment for GRPO Model Training                                                                                                                      │ │
                                                                                                                                                                  │ │
This module provides a reinforcement learning environment wrapper for AppWorld                                                                                    │ │
that can be used with GRPO (Generalized Reward-based Policy Optimization) models.                                                                                 │ │
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal
from .appworld_main import AppWorld, load_task_ids

class AppWorldEnv:

    def __init__(
        self,
        remote_environment_url: str = "http://0.0.0.0:8081",   
        ground_truth_mode: Literal["full", "minimal"] = "full",
        worker_id: str = "default",
        max_interactions: int = 30,
    ):
        self.workder_id = worker_id
        self.max_interactions = max_interactions
        self.current_step_count = 0
        self.remote_environment_url = remote_environment_url
        self.ground_truth_mode = ground_truth_mode


    def reset(self, task_id):
        self.current_step_count = 0
        
        self.env = AppWorld(
            task_id=task_id,
            experiment_name="sample",
            remote_environment_url=self.remote_environment_url,
        )
        AppWorld.init_defaults.experiment_name = f'exp_{self.workder_id}'

        obs = self.env.task.instruction
        info = {
            "task_id": task_id,
            "supervisor": dict(self.env.task.supervisor),
        }
        return obs, info


    def step(self, action) -> float:
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1
        obs = self.env.execute(action)
        done = self.env.task_completed() 
        reward = 0.0

        if self.env.task_completed():
            test_tracker = self.env.evaluate(suppress_errors=True)                                                                                                           
            reward = 1.0 if test_tracker.success else test_tracker.pass_count / test_tracker.total_count

        info = {
            "step_count": self.current_step_count
        }

        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()

class AppWorldEnvs:
    """Wrapper for multiple AppWorld environments."""
    
    def __init__(self, dataset_name="train", max_interactions=30):
        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.task_ids = load_task_ids(dataset_name)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, AppWorldEnv]:
        if self.current_index >= len(self.task_ids):
            raise StopIteration
        
        task_id = self.task_ids[self.current_index]
        self.current_index += 1
        env = AppWorldEnv(
            max_interactions=self.max_interactions,
        )
        return task_id, env


def build_appworld_envs(dataset_name="train", 
                        max_interactions=30
                    ):
    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
    )
