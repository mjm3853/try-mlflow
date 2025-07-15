from sentence_transformers import SentenceTransformer
import mlflow
import warnings
from dotenv import load_dotenv

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

example_sentences = ["A sentence to encode.", "Another sentence to encode."]

# Infer the signature of the custom model by providing an input example and the resultant prediction output.
# We're not including any custom inference parameters in this example, but you can include them as a third argument
# to infer_signature(), as you will see in the advanced tutorials for Sentence Transformers.
signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=model.encode(example_sentences),
)

# Visualize the signature
print(signature)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Introduction to Sentence Transformers")

with mlflow.start_run():
    logged_model = mlflow.sentence_transformers.log_model(
        model=model,
        name="sbert_model",
        signature=signature,
        input_example=example_sentences,
    )

inference_test = ["I enjoy pies of both apple and cherry.", "I prefer cookies."]

print(logged_model.model_uri)

# Load our custom model by providing the uri for where the model was logged.
loaded_model_pyfunc = mlflow.pyfunc.load_model(logged_model.model_uri)

# Perform a quick test to ensure that our loaded model generates the correct output
embeddings_test = loaded_model_pyfunc.predict(inference_test)

# Verify that the output is a list of lists of floats (our expected output format)
print(f"The return structure length is: {len(embeddings_test)}")

for i, embedding in enumerate(embeddings_test):
    print(f"The size of embedding {i + 1} is: {len(embeddings_test[i])}")
