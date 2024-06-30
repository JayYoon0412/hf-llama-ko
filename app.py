import argparse
import os
from time import sleep
from typing import List
from threading import Thread

import gradio as gr
from transformers import AutoTokenizer

class LLMChatHandler():
    def __init__(self, model_id: str, use_vllm: bool = False):
        self.use_vllm = use_vllm
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("")
        ]
        self.initial_instruction = (
            """
You are a chatbot designed to help users determine which hospital department they should visit based on their symptoms. Engage the user in a conversation to gather detailed information about their condition. Here are the steps to follow:

1. **Greeting and Introduction**: Start by greeting the user and introducing yourself as a chatbot here to assist them in finding the right hospital department.

2. **Symptom Inquiry**: Ask the user to describe their symptoms in detail. Encourage them to include the following information:
   - Location of pain or discomfort
   - Severity and duration of the symptoms
   - Any accompanying symptoms (e.g., fever, dizziness, nausea)
   - Any pre-existing conditions or recent events that might be related

3. **Specific Questions**: Use specific questions to get more precise information:
   - Are you experiencing chest pain? If so, for how long and how intense is it?
   - Do you have any respiratory issues, like shortness of breath or persistent coughing?
   - Are there any gastrointestinal symptoms, such as stomach pain, nausea, or vomiting?
   - Have you had any recent injuries or accidents?

4. **Additional Information**: Ask if there are any other symptoms or details they think might be relevant.

5. **Confirmation and Department Suggestion**: Based on the information provided, confirm the symptoms and suggest the appropriate hospital department.

Example conversation:
User: "I have a severe headache and nausea."
Chatbot: "I'm sorry to hear that. How long have you been experiencing these symptoms? Are there any other symptoms, such as sensitivity to light or fever?"

Remember to be empathetic and reassuring throughout the conversation. Your goal is to gather enough information to accurately direct the user to the correct hospital department.
When the user inputs the keyword "Recommend", Based on the information provided, confirm the symptoms and suggest the appropriate hospital department immediately.
너는 증상에 알맞은 병원을 추천해주는 챗봇이야
사용자는 아플 때 병원의 세부 분과, 진료과 중 어디를 선택해야할지 알고싶은데 네가 적절한 병원을 추천해줘.
현재 존재하는 진료과는 다음과 같다.
내과에는 호흡기내과, 순환기내과, 소화기내과, 혈액종양내과, 내분비대사내과, 알레르기내과, 신장내과, 감염내과, 류마티스내과, 내과(일반), 내과(입원의학)로 총 11개의 진료과가 존재한다.외과에는 간담췌외과, 대장항문외과, 이식혈관외과, 위장관외과, 유방내분비외과, 외상외과, 외과(일반), 외과(입원의학)로 총 8개의 진료과가 존재한다.그리고 내과와 외과 이외의 과의 경우 산부인과, 마취통증의학과, 비뇨의학과(비뇨기과), 신경외과, 응급의학과, 임상약리학과, 가정의학과, 방사선종양학과, 성형외과, 안과, 의공학과, 임상유전체의학과, 병리과, 신경과, 영상의학과, 이비인후과, 재활의학과, 정신과학의학과, 진단검사의학과, 흉부외과, 정형외과, 피부과, 중환자의학과, 핵의학과 등이 존재한다.
사용자는 자기 증상을 말하고, 나이와 성별을 알려줄 거야. 정보를 고려해서 추천하는 진료과를 알려줘. 
사용자의 현재 위치를 토대로 적절한 병원의 이름을 알려줘도 좋아.
사용자가 "진료과 추천 바람"이라고 작성하면 더이상의 꼬리 질문 없이 바로 진료과를 무조건 추천해줘.
"""
        )
        if use_vllm:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            engine_args = AsyncEngineArgs(
                model=model_id,
                tokenizer=None,
                tokenizer_mode="auto",
                trust_remote_code=True,
                quantization="awq" if "awq" in model_id or "AWQ" in model_id else None,
                dtype="auto",
            )
            self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            from transformers import AutoModelForCausalLM
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype="auto",
                device_map="auto")

    def chat_history_to_prompt(self, message: str, history: List[List[str]]) -> str:
        conversation = [{"role": "assistant", "content": self.initial_instruction}]
        for h in history:
            user_text, assistant_text = h
            conversation += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

        conversation.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return prompt

    async def chat_function(self, message, history):
        prompt = self.chat_history_to_prompt(message, history)
        if self.use_vllm:
            response_generator = self.chat_function_vllm(prompt)
            async for text in response_generator:
                text = self.filter_text(text)  # Filter the text
                yield text
        else:
            response_generator = self.chat_function_hf(prompt)
            for text in response_generator:
                text = self.filter_text(text)  # Filter the text
                yield text

    async def chat_function_vllm(self, prompt):
        from vllm import SamplingParams
        from vllm.utils import random_uuid
        sampling_params = SamplingParams(
            stop_token_ids=self.terminators,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2)
        results_generator = self.vllm_engine.generate(prompt, sampling_params, random_uuid())
        async for request_output in results_generator:
            response_txt = ""
            for output in request_output.outputs:
                if output.text not in self.terminators:
                    response_txt += output.text
            yield response_txt

    def chat_function_hf(self, prompt):
        from transformers import pipeline, TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tokenizer)
        pipe = pipeline(
            "text-generation",
            model=self.hf_model,
            tokenizer=self.tokenizer,
            eos_token_id=self.terminators,
            max_length=2048,  # Control maximum tokens here
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            return_full_text=False,
            streamer=streamer
        )
        t = Thread(target=pipe, args=(prompt,))
        t.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

        t.join()

    def filter_text(self, text: str) -> str:
        # Post-processing step to remove unwanted emojis and special characters
        import re
        text = re.sub(r'[^\w\s,.?!-]', '', text)  # Allow only alphanumeric characters and some punctuation
        return text

def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)

def main(args):
    print(f"Loading the model {args.model_id}...")
    hdlr = LLMChatHandler(args.model_id, args.use_vllm)

    with gr.Blocks(title=f"🤗 Chatbot with {args.model_id}", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with 🤗 {args.model_id} 🤗</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
                f"<h3>Original model: <a href='https://huggingface.co/{args.model_id}' target='_blank'>{args.model_id}</a></h3>")
        gr.ChatInterface(hdlr.chat_function)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="maywell/Llama-3-Ko-8B-Instruct", help="HuggingFace model name for LLM.")
    parser.add_argument("--port", default=7860, type=int, help="Port number for the Gradio app.")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM instead of HuggingFace AutoModelForCausalLM.")
    parser.add_argument("--tensor-parallelism", default=1, type=int, help="Number of tensor parallelism.")
    args = parser.parse_args()

    main(args)
