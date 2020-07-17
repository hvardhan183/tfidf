# TFIDF Vectorizer Service

#### TFIDF is termed as Term Frequency - Inverse Document Frequency and is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. 

## Training the vectorizer

First you need to have the latest wikipedia xml dump that can be downloaded from https://dumps.wikimedia.org/enwiki/latest/ . Look for enwiki-latest-pages-articles-multistream.xml.bz2

Once that is done, you can go ahead and run the wikiextractor

```
getwiki.sh <path-to-xml-dump>
```
This will create a `enwiki` file containing all the articles filtered!

Then you can go ahead and train the vectorizer using 

```
python3 trainvectorizer.py
```

This will take around 4-8 hours depending on the configuration of your machine. This will create two files; model.joblib and model.tar.gz

## Deploying the vectorizer on Amazon Sagemaker

The `model.joblib` can be used for vectorizing any text locally by using the following code

```
import joblib
tfidf_vectorizer = joblib.load('path/to/model.joblib')
print(tfidf_vectorizer.transform(['Any text here']))
```
The `model.tar.gz` is used to deploy it on sagemaker. You can directly put it in an S3 bucket and run the following code to deploy the model.
It is already present in `s3://sagemaker-ap-south-1-013583992408/tfidfmodel/` and you can change the path to the model accordingly.

Make sure that you've the `train.py` and `requirements.txt` while deploying the model! It takes around 6-7 minutes for the model to be deployed.

```
from sagemaker import get_execution_role
from sagemaker.sklearn  import SKLearnModel

role = get_execution_role()

sklearn_model = SKLearnModel(model_data='s3://sagemaker-ap-south-1-013583992408/tfidfmodel/model.tar.gz', role=role,
    entry_point='train.py',dependencies=['requirements.txt'],name="tfidf-vectorizer")

predictor = sklearn_model.deploy(initial_instance_count=1,
                                   instance_type='ml.t2.medium')
```

## Invoking the endpoint 

Once the endpoint is up, you can invoke the endpoint by using the following code snippet.

```
import boto3
import io
import scipy.sparse as sp

client = boto3.client('runtime.sagemaker', region_name='ap-south-1')

response = client.invoke_endpoint(EndpointName="tfidf-vectorizer", Body=json.dumps(data))
response_body = response['Body']
output=response_body.read()
out=json.loads(output)

tmp_ = io.BytesIO(out["instances"].encode('ISO-8859-1'))
vectors = sp.load_npz(tmp_)
```

####Note the output of the model is a Sparse-matrix in the form of an encoded String. Use the steps above to get the Sparse-matrix back. The length of each vector is `5010244`

You can use the `create_invoke_endpoint.py` to deploy (if the endpoint is not up) and invoke the endpoint.

## Automating the deletion of the Sagemaker endpoint

Sagemaker is a very expensive service and we incur cost based on the time the endpoint has been active. This means that the service being active 24/7 is not a good idea. At the same time we cannot create an endpoint everytime we want to use the service. So we have to have something that is cost effective and at the same time provides the required vectors with not much latency.

So, we have a lambda function that checks whether the endpoint is active and if it's active, whether it has been invoked in the last 20 mins. If an endpoint is invoked in the last 20 mins, then we'll not delete the endpoint because we might need it again in the next 20 mins. If not we take down the endpoint. This can be extended to all the sagemaker endpoints and the lambda function is written taking in view of all the endpoints. To deploy the Lambda function, go to `tfidf-service-lambda` folder. The required zip is also included as `lambda.zip` and this Lambda function is called every 20 mins to check whether to take down an endpoint. This buffer time can be adjusted in the code and we can also have a different buffer time for each sagemaker endpoint.

