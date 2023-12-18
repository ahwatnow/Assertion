from langchain.prompts     import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts     import PromptTemplate
from langchain.chains      import LLMChain
import openai
import argparse
import os

class ChatGPTAgent(object):

    def __init__(self
                 , api_key: str
                 ):
        openai.api_key     = api_key
        self.chatgpt_model = 'gpt-3.5-turbo'
        os.environ['OPENAI_API_KEY'] = openai.api_key

    def get_chatgpt_response(self, users_input: str):
        # Define the prompt template
        template = """instruction : Generate code according to the request below and provide three test cases for the generated code.
        request : {request}
        """

        # prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model_name=self.chatgpt_model) # type: ignore
        llm_chain = LLMChain(llm=model, prompt=PromptTemplate.from_template(template), verbose=True)
        output = llm_chain(inputs={"request": users_input})
        print(output)


def gen_gpt_model(api_key: str) -> LLMChain:
    # Define the prompt template
    template = """instruction : Generate code according to the request below and provide three test cases for the generated code.
    request : {request}
    """

    # prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name='gpt-3.5-turbo')
    llm_chain = LLMChain(llm=model, prompt=PromptTemplate.from_template(template), verbose=True)
    return llm_chain 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--api_key', type=str)
    parser.add_argument('-m', '--chatgpt_model', type=str,
                        default='gpt-3.5-turbo')
    config = parser.parse_args()


    users_input = input("원하시는 파이썬 코드를 설명해주세요 : ")
    agent = ChatGPTAgent(config)
    agent.get_chatgpt_response(users_input)


if __name__ == '__main__':
    main()