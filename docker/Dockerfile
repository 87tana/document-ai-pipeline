FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/

CMD ["src.serving.app.handler"]
