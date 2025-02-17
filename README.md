# AutoStar

## Purpose

The challenge we're addressing is the tendency of users to rate either 5 stars or 1 star. The objective of this project is to develop a model that can determine a more reasonable star rating based on the content of a user's review.

## Dataset

The [McDonald's Store Review](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews) dataset from Kaggle was used for this project. You can download the dataset from there.

## Jupyter Notebook: Model Training

The model was trained on Google Colab. To execute the notebook correctly, you may have to upload it to the Google Drive along with the data set. Otherwise, some minor modifications to the notebook may be required in order for everything to execute, such as the path of the CSV file and the version of Tensorflow you have installed (which should be 2.16.1). Make sure the dataset is in the same directory as the jupyter notebook and that the CSV is entitled `McDonald_s_Reviews.csv`.

## Execution and Testing of the Model

This project utilizes Docker. For more information on Docker, refer to its [documentation](https://docs.docker.com/).

There are two versions of the Docker deployment: a working version and a non-functional version.

### Working Version

The working version can be found in the `autostar-docker` directory. Once Docker is installed, you can get the project running by executing `bash redeploy.sh` in a bash terminal while inside the `autostar-docker` directory. This will deploy the model using TensorFlow in a Flask server. You should be able to interact with with the server via `http://127.0.0.1:5000` in any browser.

### Non-Functional Version with TensorFlow Serving

The non-functional version can be found in the `autostar-docker-with-TFS-NOT-WORKING` directory. Once Docker is installed, to get the project running, please execute `bash redeploy.sh` and `bash redeploy-server.sh` in a bash terminal while inside the `autostar-docker-with-TFS-NOT-WORKING` directory. This should create the Docker containers and execute both the Flask and TensorFlow Serving servers in separate containers. You should be able to interact with the Flask server via `http://127.0.0.1:5000` in any browser. They can communicate via a Docker network, but unfortunately, the TensorFlow server produces the following error:

```bash
W external/org_tensorflow/tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: FAILED_PRECONDITION: Could not find variable sequential/dense_2/kernel. This could mean that the variable has been deleted. In TF1, it can also mean the variable is uninitialized. Debug info: container=localhost, status error message=Resource localhost/sequential/dense_2/kernel/N10tensorflow3VarE does not exist.
         [[{{function_node __inference_serving_default_46412}}{{node sequential_1/dense_2_1/Cast/ReadVariableOp}}]]
```

The following steps were taken to troubleshoot it:
1. The error suggests that there might be a version mismatch between TensorFlow and TensorFlow Serving. However, both the version of TensorFlow and TensorFlow Serving were 2.16.1.
2. Alternatively, the model's architecture might be accidentally served differently from training. This seems unlikely because the architecture should be specified in the save file, but this was not completely verified, so perhaps this is the problem.
3. Re-exporting the model was attempted, but the error persisted.

## Model Monitoring

While we do enable TensorBoard through tensorboard_callback in the Jupyter notebook in the hopes of using it, we were unable to do model monitoring as the result of the previously mentioned error with the TensorFlow Serving Server.

