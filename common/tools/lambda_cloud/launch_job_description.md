I would like to build a tool that allows me to launch training jobs on a remote machine. It will take as inputs 
- a list (or single) path to a yaml config file. 
- A list of branches each yaml config should run on. Default to main. Optionally, a text file with a comma separated yaml_path, branch (or no branch), one per line. 
- A machine setup config, described below

It will produce a training job for each config on a lambda cloud machine. The lambda cloud machine will be started with the API in common/tools/lambda_cloud/lambda_api.

Once the lambda cloud machine starts, it will need to be set up by copying a number of files to the remote machine, running a few other commands on the remote machine, starting a tmux session on the remote machine, and kicking off a training job in that training session. These commands should be specified in a config file that has the following sections:

machine_types: [gpu_1x_gh200]
ssh_key: aspen
files_to_copy:
    - /home/ekf/.cache/torch/hub/checkpoints: remote_path
    - ~/.tmux.conf: ~/.tmux.conf
remote_setup_commands:
    - cp -r vigor/Chicago /tmp/
    - cp -r vigor/Seattle /tmp/
    - git clone https://github.com/ewfuentes/robot.git && cd robot && ./setup.sh
max_train_time_hours: 43

Training starts with a command along the lines of: 
bazel run //experimental/overhead_matching/swag/scripts:train -- --dataset_base /tmp/ --output_base /tmp/output --train_config path_to_train_config.yaml

A couple of key points:
Either once training is done (or otherwise exits), or the maximum time limit is reached. A shutdown procedure should start. 
- This first will rsync all files to the machine that started this process. This may be complicated as the remote machine will have to rsync data to a machine behind a jumphost with duo auth (see https://tig.csail.mit.edu/network-wireless/ssh/), or we can try dumping the data to an s3 bucket, in which case the lambda machine will need to be set up with S3 credentials. Let's try AWS. 
- Next, the machine will have to make a call through the lambda cloud API mentioned previously to terminate its own instance. This means it needs to know which one it is (maybe through IP?), and will need the API key from the host that started it. This needs to be transferred safely. 

Interaction with the remote machine should be done through ssh. First time SSHing into the device will probably have the "trust signature" yes/no prompt, which should be trusted. Accessing ssh programatically, and then entering tmux seems difficult.

We should be able to try to launch multiple jobs at once. If multiple machines are listed in the machine config, they should be attempted in order from first to last, looping back if none of them have availiable instances. Instance launching and the setup commands take some time. It would be nice if we can be running many of these in parallel processes. The Lambda API supports this with file locking.