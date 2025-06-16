from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

# Non-MoE model
runner = Runner(
    which=Which.VisionPlain(
        model_id="https://huggingface.co/Qwen/Qwen3-4B",
        arch=VisionArchitecture.Qwen3,
    ),
    in_situ_quant="Q4K",
)

# MoE model
# runner = Runner(
#     which=Which.VisionPlain(
#         model_id="https://huggingface.co/Qwen/Qwen3-30B-A3B",
#         arch=VisionArchitecture.Qwen3Moe,
#     ),
#     in_situ_quant="Q4K",
# )

messages = [
    {
        "role": "user",
        "content": "Hello! How many rs in strawberry?",
    },
]
completion = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="qwen3",
        messages=messages,
        max_tokens=1024,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
)
resp = completion.choices[0].message.content
print(resp)