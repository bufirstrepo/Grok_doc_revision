from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct", quantization="awq", tensor_parallel_size=1)  # Adjust for your GPUs

sampling_params = SamplingParams(temperature=0.0, max_tokens=300)

def grok_query(prompt: str) -> str:
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()
