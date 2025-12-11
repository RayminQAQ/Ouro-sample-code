from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig

model_name = "ByteDance/Ouro-1.4B"

config = AutoConfig.from_pretrained("ByteDance/Ouro-1.4B")
config.total_ut_steps = 3  # Use 3 recurrent steps instead of 4
config.use_cache = False
config.trust_remote_code = True

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    config=config,
    torch_dtype="auto",
)

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))