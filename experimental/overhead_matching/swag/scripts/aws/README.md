Launch Training Jobs on AWS
---------------------------

AWS Setup
=========
1. Get added to the AWS group, ask Erick for details.
2. Sign into the AWS console and on the IAM page, create a new user.
3. Add that user to the `overhead-matching` group.
4. Download the latest version of the aws cli by running:
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```
5. Create an access key for your user and enter the required details in `aws configure`.
6. If everything works, you should be able to run `aws s3 ls` and see the `rrg-overhead-matching` bucket.
