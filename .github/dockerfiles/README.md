
Running Self Hosted Github Runner
---------------------------------

To start a self-hosted github runner, cd to this directory, then run:
```
docker build -f Dockerfile.24.04 . -t github_runner:24.04
docker run github_runner:24.04
```

Inside of the docker container run:
```
./config.sh --url https://github.com/ewfuentes/robot --token ${TOKEN} --unattended
./run.sh
```
Where the value of `$TOKEN` comes from [here](https://github.com/ewfuentes/robot/settings/actions/runners/new).
