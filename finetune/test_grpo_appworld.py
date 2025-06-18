#!/usr/bin/env python3
"""
Test script for GRPO AppWorld implementation.
Tests dataset loading and reward computation without full training.
"""

import sys
sys.path.append('..')

from finetune.grpo_appworld import (
    AppWorldDataset, 
    AppWorldRewardFunction,
    syntax_validity_reward,
    api_usage_reward
)


def test_dataset_loading():
    """Test AppWorld dataset loading."""
    print("Testing AppWorld dataset loading...")
    
    try:
        # Create dataset with just 3 tasks
        dataset = AppWorldDataset(dataset_name="train", max_tasks=3)
        hf_dataset = dataset.to_dataset()
        
        print(f"✓ Successfully loaded {len(hf_dataset)} tasks")
        
        # Display first task
        if len(hf_dataset) > 0:
            first_task = hf_dataset[0]
            print("\nFirst task details:")
            print(f"  Task ID: {first_task['task_id']}")
            print(f"  Instruction: {first_task['task_instruction'][:100]}...")
            print(f"  Prompt messages: {len(first_task['prompt'])} messages")
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    return True


def test_reward_functions():
    """Test reward functions with sample completions."""
    print("\n\nTesting reward functions...")
    
    # Sample completions for testing
    test_completions = [
        [{"content": "```python\napis.supervisor.complete_task(status='success')\n```"}],
        [{"content": "Here's the code:\nimport apis\napis.execute('task')"}],
        [{"content": "invalid python syntax {"}],
        [{"content": "No code here, just text"}],
    ]
    
    # Test syntax validity reward
    print("\nSyntax validity rewards:")
    syntax_rewards = syntax_validity_reward(test_completions)
    for i, (comp, reward) in enumerate(zip(test_completions, syntax_rewards)):
        print(f"  Completion {i+1}: {reward:.2f}")
    
    # Test API usage reward
    print("\nAPI usage rewards:")
    api_rewards = api_usage_reward(test_completions)
    for i, (comp, reward) in enumerate(zip(test_completions, api_rewards)):
        print(f"  Completion {i+1}: {reward:.2f}")
    
    return True


def test_appworld_reward_function():
    """Test AppWorld environment reward function."""
    print("\n\nTesting AppWorld environment reward function...")
    
    # Note: This requires AppWorld server to be running
    print("Note: AppWorld reward function requires the AppWorld server to be running.")
    print("Without the server, the reward function will return 0.0 for all completions.")
    
    reward_func = AppWorldRewardFunction(max_interactions=10)
    
    # Test with a dummy completion
    test_completions = [
        [{"content": "```python\napis.supervisor.complete_task(status='success')\n```"}]
    ]
    test_task_ids = ["dummy_task_id"]
    
    try:
        rewards = reward_func(test_completions, [], task_id=test_task_ids)
        print(f"✓ Reward function executed successfully")
        print(f"  Reward: {rewards[0]:.2f}")
    except Exception as e:
        print(f"✗ Reward function failed: {e}")
        print("  This is expected if AppWorld server is not running")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing GRPO AppWorld Implementation")
    print("=" * 60)
    
    # Run tests
    test_dataset_loading()
    test_reward_functions()
    test_appworld_reward_function()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()