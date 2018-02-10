import argparse
import boto3
import json
import pandas as pd


s3 = boto3.client('s3')


def list_objects(bucket, prefix):
    paginator = s3.get_paginator('list_objects')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for problem in page['Contents']:
            yield problem['Key']


def read_problem(bucket, problem):
    res = s3.get_object(Bucket=bucket, Key=problem)
    body = res['Body'].read()
    if len(body) > 0:
        return json.loads(body)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket')
    parser.add_argument('output')
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()

    results = filter(None, [
        read_problem(args.bucket, problem)
        for problem in list_objects(args.bucket, args.prefix)
    ])
    dataframe = pd.DataFrame.from_records(results)
    dataframe.to_csv(args.output, index=False)
