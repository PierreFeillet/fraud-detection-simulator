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
    url="https://us-south.ml.cloud.ibm.com",
    api_key=api_key
)
project_id = "952c902e-8928-4bf6-9e60-2ed48398154e"

#model_id = "meta-llama/llama-3-3-70b-instruct"
strategy_gen_model_id = "meta-llama/llama-3-405b-instruct"
#activity_gen_model_id = "ibm/granite-13b-instruct-v2"
activity_gen_model_id = "mistralai/mistral-large"

# Adjust accrdingly
parameters_strategy = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.TEMPERATURE: 0.5,
    #GenParams.TOP_K: 50,
    GenParams.TOP_P: 0.9,
    GenParams.STOP_SEQUENCES: [
			"```end_json"
		],
}

parameters_activity = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0.4, #try 0.2
    #GenParams.TOP_K: 50,
    GenParams.TOP_P: 0.8,
    GenParams.STOP_SEQUENCES: ["```end_json"],
}
print("Available models on Watsonx:\n")
api_client = APIClient(credentials=credentials, project_id=project_id)
api_client.foundation_models.TextModels.show()
print("\n")

def watsonx_chat(prompt, model_id, parameters):
    """Sends the prompt to WatsonxLLM and returns the text response."""
    # Initialize your WatsonxLLM instance as shown in your Watson script:
    llm = WatsonxLLM(
        model_id=model_id,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id,
        params=parameters
    )
    # Create a simple prompt template that simply passes through the prompt
    prompt_template = PromptTemplate(
         input_variables=["prompt"],
         template="{prompt}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template, output_key='output')
    response = chain.invoke({"prompt": prompt})
    return response["output"]
