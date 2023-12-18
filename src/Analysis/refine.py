from __future__ import annotations
from typing     import *

import sys
sys.path.append('../')

from GenerateCode.test1 import gen_gpt_model 
from GenerateCode.test2 import *

from langchain.chains   import LLMChain # type: ignore

import argparse
from langchain.prompts     import ChatPromptTemplate    # type: ignore
from langchain.chat_models import ChatOpenAI            # type: ignore
from langchain.prompts     import PromptTemplate        # type: ignore
import openai # type: ignore

from tqdm import tqdm

import os


# config.
def get_prompt(instruction, system_prompt):
    SYSTEM_PROMPT = S_SYS + system_prompt + E_SYS
    prompt_template =  S_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful coding assistant, you always only provide answers in Python."
# config done


class DynamicAnalysis:
    
    def __init__(self
                 , api_key: str
                 , refine: str | bool
                 , max_iter: str
                 ):
        self.api_key = api_key
        if isinstance(refine, str):
            self.refine = True if refine == "True" else False 
        else:
            self.refine = refine
        self.max_iter = int(max_iter)

        # config
        openai.api_key     = api_key
        self.chatgpt_model = 'gpt-3.5-turbo'
        os.environ['OPENAI_API_KEY'] = openai.api_key

        self.chain   = self._get_llm_chain()

        # function namespace
        self.namespace = dict()


    def _get_llm_chain(self) -> LLMChain:
        model = ChatOpenAI(model_name=self.chatgpt_model) # type: ignore
        template = get_prompt(instruction, system_prompt)
        prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm_chain = LLMChain(llm=model, prompt=prompt, verbose=True, memory=memory)
        return llm_chain


    def _delete_useless(self, code: str) -> str:
        return code.strip('```').strip('python').strip('\n')


    def _gen_code(self, query: str) -> str:
        '''take a query and generate code
        '''
        req = ". Provide code only. Name of function should be `target`. Also, Do not say anything else."
        res = self.chain.predict(user_input=query + req)
        res = self._delete_useless(res)
        print(f'[RES]: {res}')
        return res
        

    def _gen_test_case(self, reentered: bool = False) -> List[Dict[str, Any]]:
        '''generate test cases about query.
        note that, you gotta define whta structure you will use to save tc.
        '''
        if not reentered:
            req = "Generate five test cases for the code snippet in the chat history. \
                Each test case should follow this format below. \
                [ \{'input': element, 'output': element\}, \{'input':..\}, ..]\
                Also, Provide code only. Do not say anything else."
        else:
            req = "Your answer is ill-formed. You MUST follow the format below.\
                  [ \{'input': element, 'output': element\}, \{'input':..\}, ..]\
                  Also, Provide code only. Do not say anything else. "
        res = self.chain.predict(user_input=req)
        processed = res.strip('```').strip('python').strip('\n')
        idx = processed.find('[')
        processed = processed[idx:]
        print(f'[TC]: {processed}')
        try: 
            tc = eval(processed)
            return tc 
        except:
            self._gen_test_case(True)


    def _execute(self, code: str, tc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        res = list() 

        exec(code, self.namespace) # def of fucntion which gpt generates
        for case in tc: 
            input = case['input']
            expected = case['output']
            target = self.namespace['target']

            # application
            if isinstance(input, tuple) or isinstance(input, list):
                temp = target(*input)
            else:
                temp = target(input) 

            # checking answer.
            if temp == expected: 
                res.append({'succ': True, 'output': temp})
            else:
                res.append({'succ': False, 'input': input, 'output': temp, 'expected': expected})
        return res


    def _check_is_passed(self, res: List[Dict[str, Any]]) -> bool:
        for elem in res:
            if not elem['succ']:
                return False
        return True


    def _get_fail_case(self) -> List[Dict[str, Any]]:
        return list(filter(lambda x: not x['succ'], self.fail_case))


    def _get_modify_command(self, fail_case: List[Dict[str, Any]]) -> str:
        basis = "Code that you generated is wrong. Give me correct code considering fail cases. Fail cases are as follows:\n"
        for case in fail_case:
            temp = f'[Input]: {case["input"]}, [Output]: {case["output"]}, [Correct answer]: {case["expected"]}\n'
            basis += temp
        return basis


    def _command_modify(self) -> str:
        fail_case = self._get_fail_case()
        query = self._get_modify_command(fail_case)
        req = "Provide code only. Name of function should be `target`. Also, Do not say anything else."
        res = self.chain.predict(user_input=query + req)
        res = self._delete_useless(res)
        print(f'[Modified]: {res}')
        return res


    def _refine(self, query: str) -> Optional[str]:
        '''do it till it reaches fixpoint 
        '''
        for cur in tqdm(range(self.max_iter)):
            if cur == 0:
                code = self._gen_code(query)
                self.tc = self._gen_test_case()
            else:
                code = self._command_modify()
            temp_res = self._execute(code, self.tc)

            if self._check_is_passed(temp_res):
                return code
            else:
                self.fail_case = temp_res
        
        return # fail case, it did not generate correct code in max iterataion.


    def get_code(self) -> Optional[str]:
        '''return clean code that we generete.
        '''
        query = input("What can I do for you? ")
        if not self.refine:
            return self._gen_code(query)

        res = self._refine(query)
        
        if res is None:
            print("[FAIL]: GPT failed in generatin correct code")
        else:
            print("[SUCC]:", res)
        return res 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key') 
    parser.add_argument('--refine', default=True)
    parser.add_argument('--max_iter', default=10)
    args = parser.parse_args()

    temp = DynamicAnalysis(args.key, args.refine, args.max_iter)
    res = temp.get_code()


if __name__ == "__main__":
    main()