import base64
import json
import boto3
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid

app = FastAPI()

# Constants
BUCKET_NAME = 'healthy-basket-image-input'
TABLE_NAME = 'healthy-basket-metadata'

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb_client = boto3.client('dynamodb')
runtime_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Uncomment this section when running locally
# session = boto3.Session(profile_name='cloudplexo-dev')
# s3_client = session.client('s3')
# dynamodb_client = session.client('dynamodb')
# runtime_client = session.client("bedrock-runtime", region_name="us-east-1")

@app.post("/analyze-food")
async def analyze_food(image: UploadFile = File(...)):
    # Read and encode the image
    image_bytes = await image.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Generate a unique ID for the image
    image_id = str(uuid.uuid4())
    
    # Upload image to S3
    s3_key = f"images/{image_id}_{image.filename}"
    s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=image_bytes)
    
    # Construct the request body for the model
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are a food analysis and health assessment assistant. Analyze the food in this image with complete factual accuracy. "
                                "Do not hallucinate or include information you're uncertain about. Provide only verifiable nutrition facts. "
                                "Create a comprehensive nutritional analysis with the following structure: "
                                
                                "1. Title: Create a brief, descriptive title for the food item. "
                                "2. Food summary: Write a short paragraph (2-3 sentences) describing the food, its key nutritional benefits, and general health impact. "
                                
                                "3. Nutritional breakdown: "
                                "   - Calories (combine value and unit like: '245kcal') "
                                "   - Protein (combine value and unit like: '12g') "
                                "   - Fat (combine value and unit like: '8g') "
                                "   - Carbohydrates (combine value and unit like: '30g') "
                                "   - Fiber (combine value and unit like: '3g') "
                                "   - Sugar (combine value and unit like: '5g') "
                                "   - Sodium (combine value and unit like: '400mg') "
                                
                                "4. Ingredients identified: List the main ingredients visible in the food. For each ingredient, provide: "
                                "   - Name of the ingredient "
                                "   - A concise description focusing on its vitamin and mineral content. For example: 'Boiled egg: Rich in vitamin B2, vitamin B12, vitamin D, selenium, and choline. Good source of complete protein.' "
                                
                                "5. Health assessment: Is this food healthy or not? Provide a brief, factual explanation. "
                                
                                "Return the result as a JSON object with exactly the following format: "
                                '{'
                                '"title": "<string>", '
                                '"food_summary": "<string>", '
                                '"nutritional_breakdown": {'
                                '  "calories": "<value>kcal", '
                                '  "protein": "<value>g", '
                                '  "fat": "<value>g", '
                                '  "carbohydrates": "<value>g", '
                                '  "fiber": "<value>g", '
                                '  "sugar": "<value>g", '
                                '  "sodium": "<value>mg"'
                                '}, '
                                '"ingredients_identified": ['
                                '  {'
                                '    "name": "<ingredient_name>", '
                                '    "nutrition_description": "<vitamin_and_mineral_content_description>"'
                                '  },'
                                '  ...'
                                '], '
                                '"Health_assessment": {"is_healthy?": <true/false>, "Reason": "<string>"}'
                                '}'
                                
                                "Ensure all numerical values are realistic and factual. If you cannot determine a precise value for any nutritional element, provide a realistic estimate based on similar foods but indicate this in your response."
                            ),
                        },
                    ],
                }
            ],
        }
    )
    
    # Invoke the model
    response = runtime_client.invoke_model(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    
    # Parse and extract the response
    response_body = json.loads(response.get("body").read())
    output_text = response_body.get('content')[0].get('text')
    
    try:
        # Find the JSON object in the response text
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        json_str = output_text[json_start:json_end]

        #nutritional_info = json.loads(output_text)
        nutritional_info = json.loads(json_str)

        # Store metadata in DynamoDB
        dynamodb_client.put_item(
            TableName=TABLE_NAME,
            Item={
                'ImageID': {'S': image_id},
                'ImageKey': {'S': s3_key},
                'NutritionalInfo': {'S': json.dumps(nutritional_info)}
            }
        )

        return JSONResponse(content=nutritional_info)

    except json.JSONDecodeError as e:
        # Handle case where output isn't proper JSON
        print(f"JSON parsing error: {e}")
        print(f"Raw output text: {output_text}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to parse model output as JSON", "details": str(e)}
        )

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)