import json
import boto3
import os
import base64
import uuid
import io 


import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import requests

from langchain import LLMChain, PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains import SequentialChain


#Fetching the bucket from environment variables
bucket_name = os.environ['BUCKET']
s3_client = boto3.client('s3', region_name='us-east-1')
s3 = boto3.resource('s3', region_name='us-east-1')
region = boto3.session.Session().region_name
rekognition = boto3.client('rekognition', region_name='us-east-1')
comprehend = boto3.client('comprehend', region_name=region)

result = {"summary": "","imageLink": "", "clickbait":"", "prompt":""}

#Defining the jumpstart deployed endpoints
sagemaker_jumpstart_summarization_endpoint = "jumpstart-dft-hf-llm-falcon-7b-instruct-bf16"
sagemaker_jumpstart_stable_diffusion_endpoint = "jumpstart-dft-stable-diffusion-v2-1-base"


#To format the input and get the output from summarization model
class ContentHandlerTextSummarization(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        result["summary"] = response_json[0]['generated_text']
        summary_text = response_json[0]['generated_text']
        return (summary_text.split(".")[0])

#Funtion to return the signed URL containing image
def get_signed_urls(images):
    s3_signed_url_list = []
    for image in images:
        #Formatting the image content
        plt.figure(figsize=(12,12))
        plt.imshow(np.array(image))
        plt.axis('off')
        formatted_image = io.BytesIO()
        plt.savefig(formatted_image, format='png')
        formatted_image.seek(0)
        output_image_name = "octank_text_to_image_app" +str(uuid.uuid4())+'.png'

        #Uploading image data into s3
        s3.Object(bucket_name, output_image_name).put(Body=formatted_image, ContentType='image/png')
        #Detecting labels in the image
        response = rekognition.detect_moderation_labels(Image={'S3Object':{'Bucket':bucket_name,'Name':output_image_name}},MinConfidence=20)
        print(response)
        if(len(response['ModerationLabels'])>0):
            s3_client.delete_object(Bucket=bucket_name, Key=output_image_name)
            result["imageLink"] = "Inappropriate elements are detected in the image. Please use the appropriate text."
            return (result["imageLink"])
        else:
            s3_signed_url_list.append(s3_client.generate_presigned_url(ClientMethod='get_object', 
                                                Params={'Bucket': bucket_name, 'Key': output_image_name}, 
                                                ExpiresIn=30000))
        result["imageLink"] = ' '.join(s3_signed_url_list) 
    return str(result["imageLink"])                                      

#To format the input and get the output from text to image model
class ContentHandlerStableDiffusion(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"prompt": prompt ,  **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        s3_signed_url_list = get_signed_urls(response_json['generated_images'])
        result['prompt'] = (response_json['prompt'])
        return s3_signed_url_list
    
#Creating Langchain LLM chain to generate summary
def get_summarization_chain(summarization_endpoint):

    content_handler = ContentHandlerTextSummarization()
    
    prompt = """{inputs}.Write a very short summary of the above text that includes the important details:
                      """
    
    prompt_template = PromptTemplate(
                            template=prompt, 
                            input_variables=["inputs"]
                        )

    click_bait_prompt = """{inputs}.Write a compelling 1 liner headline for the article above:"""
    click_bait_prompt_template = PromptTemplate(
                            template=click_bait_prompt, 
                            input_variables=["inputs"]
                        )
    
    summary_model = SagemakerEndpoint(
        endpoint_name = summarization_endpoint,
        region_name='us-east-1',
        model_kwargs= {"parameters":{"max_new_tokens":150}},
        content_handler=content_handler,
    )


    summary_chain = LLMChain(llm=summary_model,prompt=prompt_template)     
    clickbait_chain = LLMChain(llm=summary_model,prompt=click_bait_prompt_template)
    return summary_chain, clickbait_chain
 
#Creating Langchain LLM chain to generate image from text
def get_textToImage_chain(stable_diffusion_endpoint, prompt_specification):

    content_handler = ContentHandlerStableDiffusion()

    template = "{prompt_specification} of {text}"
    
    prompt = PromptTemplate(
        input_variables=["text", "prompt_specification"],
        template=template
    )
    
    texttoimage_model = SagemakerEndpoint(
        content_handler=content_handler,
        endpoint_name=stable_diffusion_endpoint,
        region_name='us-east-1',
        model_kwargs= {"num_images_per_prompt":3, "num_inference_steps":14,"height":320, "width":400, "negative_prompt":"1,pencil, of, words,cis, images, note, text, blog, article, human, text, {, }, english, 2,3,4,5,6,7,8,9,0,$,\n ,40,amazon,Convert ,people, into,an,above, Download, book, planner, know, read, write, study, article,  interactive,numbers, captions, sentences, dollars, symbols, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,writing, watermark, logos, text,words, letters, poorly Rendered face, poorly drawn face, low resolution, Images cut out at the top, left, right, bottom, blurry image, blurry image, mutated body parts"}
    )
    text_to_image_chain = LLMChain(llm=texttoimage_model, prompt=prompt,output_key="image")
    return text_to_image_chain


def lambda_handler(event, context):
    data = json.loads(json.dumps(event))

    body_input = data['body']
    jsoninputs = json.loads(body_input)

    inputs = jsoninputs['text_inputs']
    prompt_specification = jsoninputs['prompt_specification']
    # prompt_specification = data['prompt_specification']

    summary_chain, clickbait_chain = get_summarization_chain(sagemaker_jumpstart_summarization_endpoint)
    text_to_image_chain = get_textToImage_chain(sagemaker_jumpstart_stable_diffusion_endpoint, prompt_specification)

    result["clickbait"] = clickbait_chain({"inputs":inputs})['text']

    #Sequential Combination of both the chains with blogtext input
    overall_chain = SequentialChain(chains=[summary_chain, text_to_image_chain], input_variables=["inputs", "prompt_specification"] ,output_variables=["image","text"], verbose=False)
    out = overall_chain({"inputs":inputs, "prompt_specification":prompt_specification})

    print(out)
    

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Credentials": True,
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result)
    }

