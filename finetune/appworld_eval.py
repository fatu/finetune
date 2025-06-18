import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from jinja2 import Template
from appworld.task import Task, load_task_ids
from appworld.environment import AppWorld

from prompts.appworld import APPWORLD_TEMPLATE, APPWORLD_TEMPLATE_NO_HIS
from eval_utils import format_action_history, extract_code_from_response

# Initialize the Qwen3 model
# model_name = "model/Qwen/Qwen3-8B"
# print("Loading Qwen3-8B model...")
model_name = "model/Qwen/Qwen3-0.6B"
print("Loading Qwen3-8B model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,  # Use float16 for efficiency
#     device_map="auto",  # Automatically handle device placement
#     trust_remote_code=True  # Required for some Qwen models
# )

llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    # gpu_memory_utilization=0.9,
    max_model_len=24576,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=24576,
)
# print("Model loaded successfully!")

# func call model
def call_qwen_llm(messages: list[dict]) -> str:
    """
    Call Qwen3 model with a history of messages and return the response.
    """
    # Apply chat template for Qwen model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the input
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate response
    # with torch.no_grad():
    #     generated_ids = model.generate(
    #         **model_inputs,
    #         max_new_tokens=32768,  # Limit response length like the original
    #         temperature=0.1,     # Low temperature for more deterministic output
    #         do_sample=True,
    #         top_p=0.95,
    #         pad_token_id=tokenizer.eos_token_id,
    #         eos_token_id=tokenizer.eos_token_id
    #     )
    outputs = llm.generate([text], sampling_params)


    # Decode only the newly generated tokens
    # generated_ids = [
    #     output_ids[len(input_ids):] 
    #     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = outputs[0].outputs[0].text
    
    # Clean up the response
    # response = response.strip()
    
    # Remove any markdown code blocks if present
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    return response

# class Qwen React Agent
class QwenReactAgent:
    """A ReAct Agent using Qwen3-4B for AppWorld tasks with thinking tags."""

    def __init__(self, task: Task):
        self.task = task
        self.action_history: list[tuple[str, str]] = []  # (code, output) pairs
        self.step_count = 0
        self.history: list[dict] = self.initialize_conversation()

    def initialize_conversation(self) -> list[dict]:
        """Initialize the conversation with the task prompt."""
        # Create the initial prompt with task information
        prompt_dict = {
            "supervisor_first_name": self.task.supervisor.first_name,
            "supervisor_last_name": self.task.supervisor.last_name,
            "supervisor_email": self.task.supervisor.email,
            "supervisor_phone_number": self.task.supervisor.phone_number,
            "task_description": self.task.instruction,
            "step_count": 0,
            "history_length": 0,
            "action_history": "No previous actions.",
            "current_step": 1
        }
        
        # Format the prompt using Python string formatting
        initial_prompt = APPWORLD_TEMPLATE.format(**prompt_dict)
        
        return [{"role": "user", "content": initial_prompt}]

    def update_prompt_with_history(self) -> str:
        """Update the prompt with current history information."""
        history_length = len(self.action_history)  # Use all history
        
        prompt_dict = {
            "supervisor_first_name": self.task.supervisor.first_name,
            "supervisor_last_name": self.task.supervisor.last_name,
            "supervisor_email": self.task.supervisor.email,
            "supervisor_phone_number": self.task.supervisor.phone_number,
            "task_description": self.task.instruction,
            "step_count": self.step_count,
            "history_length": history_length,
            "action_history": format_action_history(self.action_history),  # Use all history
            "current_step": self.step_count + 1
        }
        
        return APPWORLD_TEMPLATE.format(**prompt_dict)

    def next_code_block(self, last_execution_output: str | None = None) -> str:
        """
        Asks Agent to generate next code block given last_execution_output and history.
        """
        if last_execution_output is not None:
            # Add the last action to history
            if self.action_history:
                # Update the last action with its output
                last_code = self.action_history[-1][0]
                self.action_history[-1] = (last_code, last_execution_output)
            
            # For subsequent steps, update the conversation
            if self.step_count > 0:
                # Add environment response
                self.history.append({"role": "assistant", "content": f"Environment:\n{last_execution_output}"})
                
                # Create new user prompt with updated history
                updated_prompt = self.update_prompt_with_history()
                self.history = [{"role": "user", "content": updated_prompt}]
        
        # Get the next response from the model
        response = call_qwen_llm(self.history)
        
        # Extract code from the response
        code = extract_code_from_response(response)
        
        # Add this action to history (output will be updated later)
        self.action_history.append((code, ""))
        self.step_count += 1
        
        # Add the full response to conversation history
        self.history.append({"role": "assistant", "content": f"You:\n{code}"})
        
        return code
        
# Test the Qwen agent on a single task using AppWorldEnv
dataset_name = "train"  # Or dev, test_normal, test_challenge
experiment_name = "qwen3_0_6b_react_agent"
max_interactions = 30
# task_id = "82e2fac_1"

task_ids = load_task_ids(dataset_name)

for index, task_id in enumerate(task_ids):
    # Load the appworld environment for the task
    with AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        remote_environment_url="http://0.0.0.0:8081",
    ) as world:
        # Load the agent with the task to solve
        print("\n\n" + "*" * 20 + f" Task {index+1}/{len(task_ids)} ({task_id})  " + "*" * 20)
        if index + 1 < 70:
            continue
        print(world.task.instruction)
        agent = QwenReactAgent(world.task)
        output: str | None = None
        # Until the task is completed or max_interactions is reached
        for _ in range(max_interactions):
            # ask the agent to generate the code block based on the history.
            code = agent.next_code_block(output)
            print("\n\n" + "%" * 20 + " CODE " + "%" * 20 + "\n" + code)
            # execute the code in the world environment
            output = world.execute(code)
            print("\n\n" + "=" * 20 + " OUTPUT " + "=" * 20 + "\n" + output)
            # Stop if task is completed
            if world.task_completed():
                print(f"\n✅ Task {task_id} completed successfully!")
                break


# # Initialize the AppWorld environment
# env = AppWorldEnv(
#     remote_environment_url="http://0.0.0.0:8081",  # Optional: use remote environment
#     max_interactions=max_interactions,
#     worker_id="qwen_test"
# )

# try:
#     # Reset environment with the task
#     obs, info = env.reset(task_id)
    
#     print("\n\n" + "*" * 20 + f" Task 1/1 ({task_id})  " + "*" * 20)
#     print(f"Instruction: {obs}")
#     print(f"Supervisor: {info['supervisor']['first_name']} {info['supervisor']['last_name']}")
#     print("-" * 60)

#     # Create task info for the agent
#     task_info = {
#         "instruction": obs,
#         "supervisor": info["supervisor"]
#     }
    
#     agent = QwenReactAgent(task_info)

#         # Until the task is completed or max_interactions is reached
#     for step in range(max_interactions):
#         print(f"\n[Step {step + 1}/{max_interactions}]")
        
#         # Ask the agent to generate the code block
#         code = agent.next_code_block(obs if step > 0 else None)
#         print("\n" + "%" * 20 + " CODE " + "%" * 20 + "\n" + code)
        
#         # Execute the code in the environment
#         obs, reward, done, step_info = env.step(code)
#         print("\n" + "=" * 20 + " OUTPUT " + "=" * 20 + "\n" + obs)
#         print(f"Reward: {reward}, Done: {done}, Won: {step_info.get('won', False)}")
        
#         # Stop if task is completed
#         if done:
#             if step_info.get('won', False):
#                 print("\n✅ Task completed successfully!")
#             else:
#                 print("\n❌ Task not completed within maximum interactions.")
#             break
#     else:
#         print("\n❌ Task not completed within maximum interactions.")

# finally:
#     # Clean up
#     pass
#     # env.close()
