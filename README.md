# **Ethereum price prediction**

## **Introduction**

The project aims to develop, train, and deploy a scalable machine learning model for predicting ETH market close values, leveraging Kubeflow's extensive capabilities. The model will be supported by a robust pipeline that enables rapid updates and iterations. Thanks to deployment as a containerized application on a Kubernetes cluster, it will ensure fault tolerance, high availability, and optimal resource management.
Moreover, the application should serve a total of 50,000 customers and during the serving time it can serve maximum 5,000 requests at the same time.

For more details: *[ml_into_the_cloud.pdf](https://github.com/Dariorab/secure_cloud_computing_project/blob/main/ml_into_the_cloud.pdf)*.

## **Tools**
- **Kubeflow**
- **Kubernetes**
- **Docker**

## **Authors and Acknowledgment**

**Ethereum price prediction** contributors:
- **[Dario Rabasca](https://github.com/Dariorab/index_repositories)**
- **[Emanuele Mancusi](https://github.com/Emancusi6)**

Thank you to all the contributors for their hard work and dedication to the project.

## **How to execute**

**Important**:
Before executing the following commands,
you need to install kubeflow and kubernets.

* Load `eth_pipeline.yaml` in the kubeflow
* Execute the training process in kublefow
* execute `streamlit run app.py`

utility commands:
```bash
kubectl delete services eth-prediction
kubectl delete deployment eth-prediction
```

## **How to load on personal docker account**

**Important**:
Before executing the following commands,
you need to install docker, kubeflow and kubernets.
### **ML models and data**
Here there are all the commands for loading all repository on personal
Docker account. After

```bash
docker build --tag dowload_data_v1 download_data/
docker tag download_data_v1 <personal_account>/download_data_v1
docker push docker.io/<personal_account>/download_data_v1 
```
```bash
docker build --tag lstm_load_data lstm_load_data/
docker tag lstm_load_data <personal_account>/lstm_load_data
docker push docker.io/<personal_account>/lstm_load_data 
```

```bash
docker build --tag load_data_v1 load_data/
docker tag load_data_v1 <personal_account>/load_data_v1
docker push docker.io/<personal_account>/load_data_v1
```

```bash
docker build --tag linear_regression_v1 linear_regression/
docker tag linear_regression_v1 <personal_account>/linear_regression_v1
docker push docker.io/<personal_account>/linear_regression_v1
```

```bash
docker build --tag random_forest_regressor_v1 random_forest_regressor/
docker tag random_forest_regressor_v1 <personal_account>/random_forest_regressor_v1
docker push docker.io/<personal_account>/random_forest_regressor_v1 
```

```bash
docker build --tag lstm lstm/
docker tag lstm <personal_account>/lstm
docker push docker.io/<personal_account>/lstm 
```

### App.py

```bash
docker build --tag eth-prediction app/
docker tag eth-prediction <personal_account>/eth-prediction
docker push docker.io/<personal_account>/eth-prediction 
```
### kubeflow

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

- [kubeflow](http://localhost:8080/)
- [kubernetes](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/)


