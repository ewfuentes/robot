
import functions_framework
import os
from google.cloud import storage
import re
import tempfile
import subprocess


storage_client = storage.Client()


def read_file(cloud_path, local_path):
    print(cloud_path)
    m = re.match(r"gs://([a-z\-_]*)/(.*)", cloud_path)
    bucket_name, file_path = m.group(1), m.group(2)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename(local_path)
    print(bucket_name, file_path)


@functions_framework.http
def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    request_json = request.get_json()
    if request.args and "message" in request.args:
        return request.args.get("message")
    elif request_json:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_input_file_name = os.path.join(temp_dir, "input.txt")
            tmp_output_file_name = os.path.join(temp_dir, "output.txt")
      #       read_file(request_json["input_path"], tmp_input_file_name)
      #       args = [
      #           "python3",
      #           "compute_hand_distribution.zip",
      #           "--input_path",
      #           tmp_input_file_name,
      #           "--output_path",
      #           tmp_output_file_name,
      #       ]
      #       subprocess.run(args)
            return f"{request_json['input_path']} {tmp_input_file_name} {tmp_output_file_name} {request_json['output_path']}"
    else:
        return f"Hello World! {os.listdir()}"
