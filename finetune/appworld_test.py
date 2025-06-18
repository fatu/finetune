from finetune.environment.appworld_main import AppWorld, load_task_ids
import csv
from datetime import datetime
import json


# Split to evaluate on.
dataset_name = "train"  # Or dev, test_normal, test_challenge

# Experiment name. Experiment outputs are store in
# experiments/outputs/{experiment_name} relative to root ("." by default)
experiment_name = "minimal_react_agent"

# CSV filename with timestamp
# csv_filename = f"appworld_train_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Define CSV headers
headers = [
    "task_id",
    "task_index",
    "instruction",
    "answer",
    "api_calls",
    "compiled_solution_code",
    "evaluation_code",
    "generator_code",
    "difficulty",
    "num_apis",
    "num_apps",
    "num_solution_code_lines",
    "required_apps",
    "required_apis",
    "full_metadata"
]

# Max number of environment interactions per task
# max_interactions = 50
# task_ids = load_task_ids("train")

# print(f"Generating CSV report for {len(task_ids)} tasks...")
# print(f"Output file: {csv_filename}")

# with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=headers)
#     writer.writeheader()
    
#     for index, task_id in enumerate(task_ids):
#         if (index + 1) % 50 == 0:
#             print(f"Processing task {index+1}/{len(task_ids)}...")
        
#         with AppWorld(
#             task_id=task_id,
#             experiment_name=experiment_name,
#         ) as world:
#             # Prepare row data
#             row = {
#                 "task_id": task_id,
#                 "task_index": index + 1,
#                 "instruction": world.task.instruction,
#                 "answer": world.task.ground_truth.answer,
#                 "api_calls": json.dumps(world.task.ground_truth.api_calls),
#                 "compiled_solution_code": world.task.ground_truth.compiled_solution_code,
#                 "evaluation_code": world.task.ground_truth.evaluation_code,
#                 "generator_code": world.task.ground_truth.generator_code,
#                 "difficulty": world.task.ground_truth.metadata["difficulty"],
#                 "num_apis": world.task.ground_truth.metadata["num_apis"],
#                 "num_apps": world.task.ground_truth.metadata["num_apps"],
#                 "num_solution_code_lines": world.task.ground_truth.metadata["num_solution_code_lines"],
#                 "required_apps": json.dumps(list(world.task.ground_truth.required_apps) if isinstance(world.task.ground_truth.required_apps, set) else world.task.ground_truth.required_apps),
#                 "required_apis": json.dumps(list(world.task.ground_truth.required_apis) if isinstance(world.task.ground_truth.required_apis, set) else world.task.ground_truth.required_apis),
#                 "full_metadata": json.dumps(world.task.ground_truth.metadata)
#             }
            
#             writer.writerow(row)

# print(f"\nCSV report generated successfully: {csv_filename}")
# print(f"Total tasks processed: {len(task_ids)}")

with AppWorld(
    task_id="82e2fac_2",
    experiment_name=experiment_name,
) as world:
    
    
    code = """
apis.supervisor.complete_task(status="fail")
    """
    # print("\n\n" + "%" * 20 + " CODE " + "%" * 20 + "\n" + code)
    # output = world.execute(code)
    # print("\n\n" + "=" * 20 + " OUTPUT " + "=" * 20 + "\n" + output)
    print(world.task_completed())
    test_tracker = world.evaluate(suppress_errors=True)

    # Check if task is completed                                                                                                            
    print(test_tracker.success)                                                                                                            
    print(test_tracker.total_count)
    print(test_tracker.num_tests)
    print(test_tracker.pass_count)
