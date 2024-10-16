import requests
import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("ENDPOINT_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")

# embeddings = AzureOpenAIEmbeddings(
#     model="text-embedding-3-large",
#     azure_endpoint=ENDPOINT,
#     api_key=API_KEY,
#     openai_api_version="2024-05-13",
# )
# input_text = "The meaning of life is 42"
# vector = embeddings.embed_query(input_text)
# print(vector[:3])
class AzureLLM:
    def __init__(self, model_name, deployment_name, api_key, endpoint, api_version):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = None
        self.deployment_name = deployment_name
        self.api_version = api_version
    
    def connect(self):
        self.headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key
        }
        print(f"Connected to Azure OpenAI model: {self.model_name} at {self.endpoint}")

    def generate_text(self, prompt):
        data = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
        }
        
        try:
            # Make a POST request to the Azure OpenAI API
            response = requests.post(
                f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}",
                headers=self.headers,
                json=data
            )
            # Raise an exception if the response contains an HTTP error
            response.raise_for_status()
            
            # Extract the text from the API response
            result = response.json()
            return result['choices'][0]['text'].strip()

        except requests.exceptions.RequestException as e:
            # Handle any exceptions that occur during the API call
            print(f"Error generating text with Azure OpenAI: {str(e)}")
            return None

    def disconnect(self):
        self.headers = None
        print(f"Disconnected from Azure OpenAI model: {self.model_name}")

azure_llm = AzureLLM(MODEL_NAME, DEPLOYMENT_NAME, API_KEY, ENDPOINT, API_VERSION)

azure_llm.connect()

output = azure_llm.generate_text("Hi how are you?")
print(output)

azure_llm.disconnect()