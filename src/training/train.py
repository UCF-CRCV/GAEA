import os
import ast
import pathlib
from types import GeneratorType

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl

from training.trainer import QwenTrainer
from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward



local_rank = None

def view_trainable_params(model):
    """
    Print all trainable parameters in the model.
    Used for debugging and understanding which parts of the model are being fine-tuned.
    
    Args:
        model: The PyTorch model
    """
    print('Parameters set to Train')
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)
    print()


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def check_parameter_type(param_iterator):
    """
    Check if the iterator contains named parameters or just parameters.
    
    Args:
        param_iterator: Iterator over model parameters
    
    Returns:
        str: Type of parameters ("Empty", "named_parameters", or "parameters")
    """
    first_item = next(iter(param_iterator), None)
    if first_item is None:
        return "Empty"
    if isinstance(first_item, tuple) and len(first_item) == 2 and isinstance(first_item[0], str):
        return "named_parameters"
    if torch.is_tensor(first_item):
        return "parameters"

def set_requires_grad(parameters, requires_grad, debug=False):
    """
    Set requires_grad flag for all parameters in the iterator.
    
    Args:
        parameters: Iterator over model parameters
        requires_grad (bool): Value to set for requires_grad
        debug (bool): Whether to print debug information
    """
    parameter_type = check_parameter_type(parameters)
    # Handle different types of parameter iterators
    if parameter_type == 'named_parameters':
        for n, p in parameters:
            p.requires_grad = requires_grad
            if debug:
                print(f"{n} requires_grad? = {requires_grad}")
                print()
        if debug: breakpoint() 
    else:
        for p in parameters:
            p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    """
    Configure the vision tower based on training arguments.
    
    Args:
        model: The model to configure
        training_args: Training arguments
        compute_dtype: Computation data type
        device: Device to use
    """
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.named_parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
def configure_mlp(model, training_args, compute_dtype, device):
    """
    Configure the MLP components of the model based on training arguments.
    
    Args:
        model: The model to configure
        training_args: Training arguments
        compute_dtype: Computation data type
        device: Device to use
    """
    merger_params = model.visual.merger.named_parameters()
    set_requires_grad(merger_params, training_args.tune_merger, debug=False)

    if training_args.finetune_liger_norm:
        model.visual.merger.ln_q.weight.requires_grad = True


def configure_llm(model, training_args):
    """
    Configure the language model components based on training arguments.
    
    Args:
        model: The model to configure
        training_args: Training arguments
    """
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Liger-kernel for Qwen2.5 is not supported yet.
    replace_qwen2_5_with_mixed_modality_forward()

    
    # Validate arguments
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        # Handle LoRA configuration
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    # Set up local rank for distributed training
    local_rank = training_args.local_rank
    
    # Set data type
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    # Load the appropriate model based on model ID
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
        **bnb_model_from_pretrained_args
    )


    # Disable caching during training
    model.config.use_cache = False
    
    # Configure model components
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    # Configure gradient checkpointing if enabled
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Add LoRA adapter if enabled
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)
        # Configure MLP after LoRA to ensure proper parameter freezing
        configure_mlp(model, training_args, compute_dtype, training_args.device)

    # Set up processor with proper configuration
    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                            # The default setting is padding_side="left"
                                            # When training using the right-side padding is more efficient
                                              padding_side="right",
                                              min_pixels=data_args.min_pixels,
                                              max_pixels=data_args.max_pixels)



    # Set model config from processor settings
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    # Handle mixed precision and normalization layers
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Create dataset and data loaders
    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    # Initialize trainer
    trainer = QwenTrainer(
        model=model,
        args=training_args,
        **data_module
    )

    # Show trainable parameters if set to True
    if training_args.view_trainable_params:
        view_trainable_params(model)

    # Start training, resume if checkpoint exists
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final state
    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()