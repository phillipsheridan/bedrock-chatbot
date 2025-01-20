# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json
import os

from botocore.exceptions import ClientError

#env permissions
os.environ["AWS_PROFILE"] = "philsher"

data = None
with open('vars.json', 'r') as f:
    # Load the JSON data into a Python dictionary
    data = json.load(f)


# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")


# Format the request payload using the model's native structure.
native_request = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": data["prompt"]}],
        }
    ],
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=data["model_id"], body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{data["model_id"]}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["content"][0]["text"]
print(response_text)