from openai import OpenAI
from treeQA_Config import (
    model_name,
    aliApiKey,
    deepseekApiKey,
    groqApiKey,
    groqModel,
    openaiApiKey,
)


def get_groq_response(prompt, query):
    client = OpenAI(
        api_key=groqApiKey,
        base_url="https://api.groq.com/openai/v1",
        max_retries=5,
    )
    completion = client.chat.completions.create(
        model=groqModel,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    if not completion.choices or completion.choices[0].message.content is None:
        raise RuntimeError("Groq returned an empty chat completion.")
    return completion.choices[0].message.content, completion.usage.total_tokens


def get_DeepSeek_Response(prompt,query):

    client = OpenAI(api_key=deepseekApiKey, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
        max_tokens=2048,
        temperature=0.0
    )

    currentTokenCount = response.usage.total_tokens
    return response.choices[0].message.content, currentTokenCount

def get_deepseekV3(prompt, query):
    api_key = aliApiKey
    client = OpenAI(

        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="deepseek-v3",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ],
        max_tokens=4096
    )

    currentTokenCount = completion.usage.total_tokens


    
    return completion.choices[0].message.content,currentTokenCount
def get_qwen14b_ali(prompt, query):
    api_key = aliApiKey
    client = OpenAI(

        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="qwen2.5-14b-instruct",  
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ],
        max_tokens=4096
    )

    currentTokenCount = completion.usage.total_tokens


    
    return completion.choices[0].message.content,currentTokenCount


def get_gpt_response(prompt,query):
    client = OpenAI(api_key=openaiApiKey or None)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.01
    )

    currentTokenCount = completion.usage.total_tokens


    
    return completion.choices[0].message.content,currentTokenCount

model_functions = {
    "groq": get_groq_response,
    "deepseekV3-chat": get_DeepSeek_Response,
    "qwen2.5-instruct-14b": get_qwen14b_ali,
    "deepseekV3_ali": get_deepseekV3,
    "gpt3.5-turbo":get_gpt_response
}
def getModelResponse(prompt, query, model_name=model_name):
    response_function = model_functions.get(model_name)
    if response_function:
        return response_function(prompt, query)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

if __name__ == "__main__":
    print(getModelResponse("You are a helpful assistance.", "Who are you?"))
