#!/usr/bin/env python3
"""
AppWorld Trajectory Data Extractor

This script helps extract conversational training data from AppWorld trajectories.
"""

import json
import base64
from typing import Dict, Any
import requests
import argparse
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

task_ids = ['6bdbc26_3', '37a8675_2', 'd4e9306_3', 'df61dc5_1', '57c3486_1', '6c2c621_3', '4ec8de5_3', '530b157_1', '396c5a2_1', '4ec8de5_2', '6c2c621_2', 'd4e9306_2', '37a8675_3', '6bdbc26_2', '6171bbc_1', '68ee2c9_3', '3ab5b8b_2', '0d8a4ee_3', '4fab96f_2', '383cbac_2', '23cf851_1', 'fac291d_1', '383cbac_3', '50e1ac9_1', '4fab96f_3', '3ab5b8b_3', '0d8a4ee_2', '68ee2c9_2', 'b119b1f_1', '50e1ac9_2', '0d8a4ee_1', 'b119b1f_2', '68ee2c9_1', '6171bbc_3', '23cf851_3', 'fac291d_2', 'fac291d_3', '383cbac_1', '23cf851_2', '6171bbc_2', 'b119b1f_3', '3ab5b8b_1', '4fab96f_1', '50e1ac9_3', '57c3486_3', 'df61dc5_3', 'd4e9306_1', '6bdbc26_1', '396c5a2_2', '530b157_3', '4ec8de5_1', '6c2c621_1', '530b157_2', '396c5a2_3', '37a8675_1', '57c3486_2', 'df61dc5_2', '34d9492_1', '6104387_1', '302c169_1', '692c77d_1', '771d8fc_1', '76f2c72_1', '287e338_1', 'cf6abd2_1', '82e2fac_1', 'ce359b5_1', 'c901732_1', '2a163ab_1', 'aa8502b_1', '60d0b5b_1', 'b0a8eae_1', 'ccb4494_1', '27e1026_1', 'e7a10f8_1', '229360a_1', '7d7fbf6_1', '07b42fd_1', 'e3d6c94_1', 'b7a9ee9_1', '6ea6792_1', '3c13f5a_1', 'e85d92a_1', '22cc237_1', 'd0b1f43_1', '29caf6f_1', 'afc0fce_1']


def decrypt_trajectory(encrypted_data: str, secret_key: str = "mysecretkey12345") -> str:
    """
    Decrypt AppWorld trajectory data using AES decryption.
    """
    try:
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        # Extract IV (first 16 bytes) and ciphertext
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        
        # Create cipher and decrypt
        key = secret_key.encode('utf-8')[:16].ljust(16, b'\0')  # Ensure 16 bytes
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(ciphertext)
        
        # Remove padding
        decrypted = unpad(decrypted, AES.block_size)
        
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Decryption failed: {e}")
        return ""


def extract_appworld_trajectory(task_id: str) -> Dict[str, Any]:
    """
    Extract and decrypt AppWorld trajectory data.
    """
    try:
        # Fetch encrypted trajectory
        url = f"https://appworld.dev/trajectories/{task_id}.txt"
        response = requests.get(url)
        response.raise_for_status()
        
        # Decrypt the data
        decrypted_data = decrypt_trajectory(response.text)
        if not decrypted_data:
            return {}
        
        # Parse JSON
        trajectory_data = json.loads(decrypted_data)
        return trajectory_data
    
    except Exception as e:
        print(f"Error extracting trajectory {task_id}: {e}")
        return {}


def get_system_prompt() -> str:
    """
    Get the AppWorld system prompt template.
    """
    return """I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.
18. Please make sure you include some runnable code in the code blocks. don't just output comment only
19. Make sure you don't try the same code again and again which give you the same result"""


def format_for_training(trajectory: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Format trajectory data for conversational AI training.
    """
    if not trajectory:
        return {
            "task_id": task_id,
            "instruction": "",
            "conversations": [],
            "metadata": {
                "evaluation": "",
                "supervisor": "",
                "agent": ""
            }
        }
    
    training_data = {
        "task_id": task_id,
        "instruction": trajectory.get("instruction", ""),
        "conversations": []
    }
    
    # Add system prompt as first message
    training_data["conversations"].append({
        "role": "system",
        "content": get_system_prompt()
    })
    
    # Add task instruction as second message with template format
    if trajectory.get("instruction"):
        task_prompt = f"""Using these APIs, now generate code to solve the actual task:

My name is: {{{{ supervisor.first_name }}}} {{{{ supervisor.last_name }}}}. My personal email is {{{{ supervisor.email }}}} and phone number is {{{{ supervisor.phone_number }}}}.

Task:

{trajectory["instruction"]}"""
        training_data["conversations"].append({
            "role": "user",
            "content": task_prompt
        })
    
    # Format messages for training
    messages = trajectory.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("message", "")  # Use "message" key instead of "content"
        
        if role == "agent":
            training_msg = {
                "role": "assistant",
                "content": content
            }
        else:
            # User, system, or environment messages
            training_msg = {
                "role": "user",
                "content": content
            }
        
        training_data["conversations"].append(training_msg)
    
    # Add metadata
    evaluation = trajectory.get("evaluation", {})
    supervisor = trajectory.get("supervisor", {})
    
    training_data["metadata"] = {
        "evaluation": evaluation.get("status", "") if isinstance(evaluation, dict) else str(evaluation),
        "supervisor": supervisor.get("name", "") if isinstance(supervisor, dict) else str(supervisor),
        "agent": "AppWorld Agent",
        "datetime": trajectory.get("datetime", ""),
        "evaluation_report": evaluation.get("report", "") if isinstance(evaluation, dict) else ""
    }
    
    return training_data




def extract_all_trajectories():
    """
    Extract training data for all predefined task IDs.
    """
    all_training_data = []
    successful_extractions = 0
    failed_extractions = 0
    
    print(f"Starting batch extraction for {len(task_ids)} task IDs...")
    
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] Processing task: {task_id}")
        
        try:
            # Extract trajectory data
            trajectory = extract_appworld_trajectory(task_id)
            
            if trajectory:
                # Format for training
                training_data = format_for_training(trajectory, task_id)
                all_training_data.append(training_data)
                successful_extractions += 1
                print(f"  ✓ Success - {len(training_data['conversations'])} conversations")
            else:
                failed_extractions += 1
                print(f"  ✗ Failed - No trajectory data")
        
        except Exception as e:
            failed_extractions += 1
            print(f"  ✗ Failed - Error: {e}")
    
    print(f"\nBatch extraction complete:")
    print(f"  Successful: {successful_extractions}")
    print(f"  Failed: {failed_extractions}")
    print(f"  Total training samples: {len(all_training_data)}")
    
    return all_training_data


def main():
    parser = argparse.ArgumentParser(description="Extract AppWorld trajectory data for training")
    parser.add_argument("--task-id", help="Single AppWorld task ID (e.g., b7a9ee9_1)")
    parser.add_argument("--all", action="store_true", help="Extract all predefined task IDs")
    parser.add_argument("--output", default="trajectory_training_data.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.all:
        # Extract all trajectories
        all_training_data = extract_all_trajectories()
        
        # Save combined data
        with open(args.output, 'w') as f:
            json.dump(all_training_data, f, indent=2)
        
        print(f"\nSaved {len(all_training_data)} training samples to {args.output}")
        
        # Print summary statistics
        total_conversations = sum(len(data['conversations']) for data in all_training_data)
        print(f"Total conversations across all samples: {total_conversations}")
        
    elif args.task_id:
        # Extract single trajectory
        print(f"Extracting trajectory for task: {args.task_id}")
        
        # Extract trajectory data
        trajectory = extract_appworld_trajectory(args.task_id)
        
        # Format for training
        training_data = format_for_training(trajectory, args.task_id)
        
        # Save to JSON
        with open(args.output, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Saved training data to {args.output}")
        print(f"Total conversations: {len(training_data['conversations'])}")
        print(f"Evaluation: {training_data['metadata']['evaluation']}")
    
    else:
        print("Please specify either --task-id or --all")
        return


if __name__ == "__main__":
    main()