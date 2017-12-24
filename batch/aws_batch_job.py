import argparse
import boto3
import os
import datetime


batch = boto3.client('batch')
s3 = boto3.client('s3')


def problem_name_from_path(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]


def list_input_problems(bucket):
    paginator = s3.get_paginator('list_objects')
    for page in paginator.paginate(Bucket=bucket, Prefix='osil/'):
        for input_problem in page['Contents']:
            name = input_problem['Key']
            if name.endswith('.osil'):
                yield name


def get_sol(bucket, problem_name):
    sol_name = 'sol/' + problem_name + '.p1.sol'
    try:
        res = s3.head_object(Bucket=bucket, Key=sol_name)
        return sol_name
    except:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-bucket', required=True)
    parser.add_argument('--output-bucket', required=True)
    parser.add_argument('--job-definition', required=True)
    parser.add_argument('--job-queue', required=True)
    args = parser.parse_args()

    problems = list_input_problems(args.input_bucket)
    for problem in problems:
        problem_name = problem_name_from_path(problem)
        sol = get_sol(args.input_bucket, problem_name)
        now = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
        jobname = 'minlplib2-' + problem_name + '-' + now
        if sol is not None:
            command = [
                '-p', 's3://' + args.input_bucket + '/' + problem,
                '-s', 's3://' + args.input_bucket + '/' + sol,
                '-o', 's3://' + args.output_bucket + '/' + problem_name + '.json'
            ]
        else:
            command = [
                '-p', 's3://' + args.input_bucket + '/' + problem,
                '-o', 's3://' + args.output_bucket + '/' + problem_name + '.json'
            ]
        print('Submitting job: {}'.format(command))
        batch.submit_job(
            jobName=jobname,
            jobQueue=args.job_queue,
            jobDefinition=args.job_definition,
            containerOverrides={
                'command': command
            }
        )
