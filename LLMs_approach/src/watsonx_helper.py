from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai import Credentials

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com/",
    api_key=api_key
)
project_id = "caa5caee-5ea4-4604-b458-2d1dd74e8ce2"

model_id_1 = "meta-llama/llama-3-3-70b-instruct"

# Adjust accrdingly
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 2000,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
    GenParams.STOP_SEQUENCES: [
			"```end_json"
		],
}

api_client = APIClient(credentials=credentials, project_id=project_id)

print("Available models on Watsonx:\n")
api_client.foundation_models.TextModels.show()
print("\n")
print(f"Using model: {model_id_1}")

# Initialize your WatsonxLLM instance as shown in your Watson script:
llm1 = WatsonxLLM(
    model_id=model_id_1,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params=parameters
)

def watsonx_chat(prompt):
    """Sends the prompt to WatsonxLLM and returns the text response."""
    # Create a simple prompt template that simply passes through the prompt
    prompt_template = PromptTemplate(
         input_variables=["prompt"],
         template="{prompt}"
    )
    chain = LLMChain(llm=llm1, prompt=prompt_template, output_key='output')
    response = chain.invoke({"prompt": prompt})
    return response["output"]
