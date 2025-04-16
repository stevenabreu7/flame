"""
uv run python test_hf_model.py --model_name=stevenabreu7/hgrn-sparse90-340M-10B-20k --prompt="Hello my name is "
"""
import argparse
import torch
import fla
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_model(model_name, prompt, max_length=100, temperature=0.7):
    # Load model and tokenizer from Hugging Face
    print(f"Loading model and tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"           # Automatically choose best device
    )
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    print(f"Generating text for prompt: {prompt}")
    with torch.no_grad():
        breakpoint()
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
        )
    
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel response:")
    print(response)
    
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a Hugging Face model")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="HF model name or path (e.g., 'username/model-name')")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to test the model")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    
    args = parser.parse_args()
    
    test_model(args.model_name, args.prompt, args.max_length, args.temperature)
