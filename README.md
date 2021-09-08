# Meta-learning codebase

## Env setup
In order to use the code build the docker from the repository folder:
```Shell
docker build -t meta_learning:v1.0 -f docker/Dockerfile .
``` 
or make the conda environment:
```Shell
conda env create -f docker/conf_files/metalearning_pytorch.yml
```

## Dataset setup 
In order to use docker the dataset paths should be set as env variables (either manually or in _docker/run_docker.sh_)
then run _run_docker.sh_.

The folder structure should be the following (running _run_docker.sh_ will automatically make this structure according 
to _docker-compose.yml_):
```Shell
├── parent_folder
    ├── meta_learning
        ├── {the code ...}      
    ├── data
        ├── imagenet
        ├── imagenet64
        ├── {other datasets ...}
    ├── results
        ├── {experiment result folders will be saved here}
```
  