import os
import json
import argparse
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


def extract_arguments() -> argparse.Namespace:
    """Extract command line arguments."""
    argparser = argparse.ArgumentParser(
        description="Finetuning script for model training."
    )
    argparser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the experiment files.",
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name of the model to finetune.",
    )
    argparser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for the model.",
    )
    argparser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load the model in 4-bit precision.",
    )
    return argparser.parse_args()


def preprocess_function(example: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "completion": [{"role": "assistant", "content": example["answer"]}],
    }


def create_bnb_config(load_in_4bit: bool):
    """Create BitsAndBytesConfig for quantization."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


def setup_model_and_tokenizer(model_name: str, max_seq_length: int, load_in_4bit: bool):
    """Setup model and tokenizer with optional quantization and LoRA."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create quantization config if needed
    bnb_config = create_bnb_config(load_in_4bit)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        trust_remote_code=True,
    )

    # Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=False,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model, tokenizer


def finetune_model() -> None:
    """Finetune the model."""
    # Extract Arguments
    args = extract_arguments()
    EXPERIMENT_PATH: str = args.experiment_path
    MODEL_NAME: str = args.model_name
    MAX_SEQ_LENGTH: int = args.max_seq_length
    LOAD_IN_4BIT: bool = args.load_in_4bit

    # Path Creation
    DATASET_PATH: str = os.path.join(EXPERIMENT_PATH, "datasets")
    CHECKPOINT_PATH: str = os.path.join(EXPERIMENT_PATH, "models", "checkpoints")

    assert os.path.exists(DATASET_PATH), f"Missing datasets (path={DATASET_PATH})."
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Model Setup
    model, tokenizer = setup_model_and_tokenizer(
        MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT
    )

    # Load Dataset and Preprocess
    TEST_DATASET_PATH = os.path.join(DATASET_PATH, "train.jsonl")
    train_dataset = datasets.load_from_disk(TEST_DATASET_PATH)
    train_dataset = train_dataset.map(
        preprocess_function, remove_columns=["question", "answer", "node_id"]
    )

    # Finetune the Model
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            max_steps=30,  # TODO: Remove
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to="none",  # TODO: Change to Wandb
            output_dir=CHECKPOINT_PATH,
            save_strategy="epoch",
            bf16=not LOAD_IN_4BIT,
            fp16=LOAD_IN_4BIT,
        ),
    )

    trainer.train()

    # Save the Model
    os.makedirs(os.path.join(EXPERIMENT_PATH, "models"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_PATH, "models", "sft_model"), exist_ok=True)

    # Save the LoRA adapter
    model.save_pretrained(os.path.join(EXPERIMENT_PATH, "models", "sft_model"))
    tokenizer.save_pretrained(os.path.join(EXPERIMENT_PATH, "models", "sft_model"))


if __name__ == "__main__":
    finetune_model()
