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
from appworld import AppWorld, load_task_ids

class AppWorldEnv:

    def __init__(
        self,
        remote_environment_url: str = "http://0.0.0.0:8081",   
        ground_truth_mode: Literal["full" "minimal"] = "full",
        worker_id: str = "default",
        max_interactions: int = 20,
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


    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1

        obs = self.env.execute(action)

        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        reward = 10.0 if self.env.task_completed() else 0.0

        info = {
            "won": self.env.task_completed(),
            "step_count": self.current_step_count
        }

        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()