from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-26b-a4b-it",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
)