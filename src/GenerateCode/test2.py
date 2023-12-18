"""
pip -q install git+https://github.com/huggingface/transformers
pip install -q datasets loralib sentencepiece
pip -q install bitsandbytes accelerate xformers einops
pip -q install langchain
"""
from __future__ import annotations
from typing     import *

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from huggingface_hub import login
import argparse
import json
import textwrap

def get_prompt(instruction, system_prompt):
    SYSTEM_PROMPT = S_SYS + system_prompt + E_SYS
    prompt_template =  S_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text

S_INST, E_INST = "[INST]", "[/INST]"
S_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful coding assistant, you always only provide answers in Python."


def gen_chain() -> LLMChain:
    '''Be careful of using this function. it includes personal information. 
    '''
    access_token = "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"
    model_name   = "codellama/CodeLlama-7b-Instruct-hf"
    login(token=access_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name,)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map = 'auto',
                                             torch_dtype=torch.float16
                                             )
    # model.config.use_cache = True
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map = 'auto',
                    max_new_tokens = 512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    )

    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    llm_chain = LLMChain(llm=llm,
                        prompt=prompt,
                        verbose=True,
                        memory=memory,
                        )
    return llm_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--access_token', type=str, default = "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")
    parser.add_argument('-m', '--model_name', type=str,
                        default='codellama/CodeLlama-7b-Instruct-hf')
    args = parser.parse_args()

    login(token=args.access_token)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             device_map = 'auto',
                                             torch_dtype=torch.float16,
                                             #load_in_8bit=True,
                                             )
    # model.config.use_cache = True
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map = 'auto',
                    max_new_tokens = 512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    )

    template = get_prompt(instruction, system_prompt)
    prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    llm_chain = LLMChain(llm=llm,
                        prompt=prompt,
                        verbose=True,
                        memory=memory,
                        )

    users_input = input("How may I help you with coding?")
    print(llm_chain.predict(user_input=users_input+". Provide code only. Do not say anything else."))
    print("\nError Case Generating...\n")
    print(llm_chain.predict(user_input="Generate test cases for the code snippet in the chat history."))
    print("\nError Case Generated.\n")

if __name__ == '__main__':
    main()

