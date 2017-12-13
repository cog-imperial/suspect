SUSPECT: Batch Jobs
===================

The project contains a `Dockerfile` to build a container running the `model_summary.py` script.

Setup:

 * Create a new ECS Repository for `suspect`.
 * Follow the instructions to build the `supect` image and publishing on ECS
 * Create a new AWS Batch job definition (you can use `job_definition.json` as a template)
 * Create a new AWS Batch queue and compute environment

Prepare your files:

 * Create an S3 bucket with your input problems:
   - 'osil/' folder contains the problems
   - 'sol/' folder contains the solution files (optional)
 * Create an S3 bucket for output


Finally to submit the jobs, run the provided `aws_batch_job.py` script:

    python batch/aws_batch_job.py \
	    --input-bucket my-input-bucket \
		--output-bucket my-output-bucket \
		--job-definiton "JOB_DEFINITION_ARN" \
		--job-queue "QUEUE_NAME"

You can check the status of your jobs in the AWS Batch Dashboard, you
will find the results in the output bucket once the jobs complete.
