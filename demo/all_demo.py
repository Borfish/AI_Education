import os
import platform
from transformers import AutoTokenizer, AutoModel
from demo_get_composition import ocr_write

my_model_path = '..//..//model//ChatGLM3-main//model'
MODEL_PATH = os.environ.get('MODEL_PATH', my_model_path)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    change_str = ocr_write(model, tokenizer, 'demo2.jpg', flag=1, save = 0)
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    query = "你是英语老师，我想和你讨论我这篇作文写的如何：" + change_str
    while True:
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")
        query = input("\n用户：")


if __name__ == "__main__":
    main()