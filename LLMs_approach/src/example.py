
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods



from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

credentials = Credentials(
    url="https://eu-gb.ml.cloud.ibm.com",
    api_key=api_key
)

project_id = "cd999bb0-42e0-413f-a706-9d92633e1c38"

api_client = APIClient(credentials=credentials, project_id=project_id)
api_client.foundation_models.TextModels.show()

model_id_1 = "meta-llama/llama-3-3-70b-instruct"

# Adjust accrdingly
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 2000,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

llm1 = WatsonxLLM(
    model_id=model_id_1,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params=parameters
    )

llm1.dict()

prompt_1 = PromptTemplate(
    input_variables=["topic"], 
    template="Generate a random question about {topic}: Question: "
)


chain1 = LLMChain(llm=llm1, prompt=prompt_1, output_key='question')

life =  "life"
result = chain1.invoke({"topic": life})
print(result)
