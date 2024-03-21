import easyocr
import os
import platform
from transformers import AutoTokenizer, AutoModel

stop_stream = False


def ocr_write(image_path='', path='./', name='demo.txt', flag=0):
    reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
    try:
        result = reader.readtext(image_path, detail=0)
    except (IOError):
        print('找不到图片:%s'%image_path)

    out_str = ' '.join(result)

    out_path = path + name
    print('正在生成原文')
    if len(out_str) != 0:
        with open(out_path, 'w') as f:
            f.write(out_str)

    if(flag):
        my_model_path = '..//..//model//ChatGLM3-main//model'
        MODEL_PATH = os.environ.get('MODEL_PATH', my_model_path)
        TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

        os_name = platform.system()
        # clear_command = 'cls' if os_name == 'Windows' else 'clear'

        welcome_prompt = "正在对可能的错字进行处理"
        past_key_values, history = None, []
        global stop_stream
        print(welcome_prompt)
        out_path = path + 'alter_' + name

        alter_str = ''
        query = '你是英语老师，修改这篇英语作文错误的单词字母，仅修改单词即可：' + out_str

        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                # print(response[current_length:], end="", flush=True)
                alter_str = alter_str + response[current_length:]
                current_length = len(response)
        if len(alter_str) != 0:
            with open(alter_str, 'w') as f:
                f.write(alter_str)
            print("修改后完成")

def main():
    # ocr_write('./demo2.jpg')
    ocr_write('./demo2.jpg',flag=1)

if __name__ == "__main__":
    main()

