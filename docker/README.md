### prepare docker for project:
  - git clone the code
  - make new directory for your project at  docker/{your project name}
  - copy the base run scripths from  docker/base  to your project's directory as  run_docker_{project}_{machine}.sh
  - change the dataset and results paths in the scripts
  - run docker with the script
  - export CUDA_VISIBLE_DEVICES={decive number}
  - start training with  *python main.py args*  in the  {code}-copy  directory
    or start pycharm with  pycharm.sh  and start coding in the  {code}  directory

  !!! train only with the code in  {code}-copy  directory and code in the  {code}  directory !!!
  !!! the copy directory is only existing in the docker and will be deleted with the docker container !!! 

#### paths:
* KITTI:
	* dgx:  
	  > export RAW_PATH=/raid/data/kitti_raw/	  
	  > export DEPTH_PATH=/raid/data/kitti_depth/	  
	  > export RGB_PATH=/raid/pointcloud/self-supervised-depth-completion/kitti-rgb/ 
  
	+ workstation:  
	  > export RAW_PATH=/data_ssd/uia94835/kitti_raw		  
	  > export DEPTH_PATH=/data_ssd/uia94835/kitti_depth		  
	  > export RGB_PATH=/data_ssd/uia94835/kitti_rgb-self-supervised-depth-completion/kitti-rgb 

#### detach from docker:
	ctlr+p ctlr+q
	
#### return to docker:
	docker attach <container id>
