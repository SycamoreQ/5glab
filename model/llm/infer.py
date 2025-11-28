from ray.data.llm import vLLMEngineProcessorConfig , build_llm_processor

model_source='Qwen/Qwen2.5-32B-Instruct' 

config = vLLMEngineProcessorConfig(
    model_source=model_source,
    accelerator_type='L4',
    engine_kwargs={
        'max_num_batched_tokens': 8192,
        'max_model_len': 8192,
        'max_num_seqs': 128, 
        'tensor_parallel_size': 4,
        'trust_remote_code': True,
    },
    concurrency=1,
    batch_size=128,  
)


# Build the processor using a preprocessor that uses the generated prompt.
processor = build_llm_processor(
    config,
    preprocess=lambda row: dict(
        messages=[
            {"role": "user", "content": row["prompt"]},
        ],
        sampling_params=dict(
            temperature=0,
            max_tokens=1024, # max reponse tokens is 1024
            detokenize=False,
        ),
    ),
    postprocess=lambda row: dict(
        resp=row["generated_text"],
        **row,  # Return all original columns.
    ),
)

# Process the dataset using the LLM inference processor.
ds = processor(ds)
results = ds.take_all()


# Print the output for each row.
for row in results:
    print(row)