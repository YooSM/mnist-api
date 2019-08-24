# MNIST classification by TensorFlow #

- [MNIST For ML Beginners](https://www.tensorflow.org/tutorials/mnist/beginners/)
- [Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)

This Rest API related to
- [MNIST Web Application](https://github.com/Writtic/mnist-web-app)

### Requirement ###

- Python 3.6.5
  - TensorFlow 1.12.0


### How to run ###

    $ pip install -r requirements-mac.txt # for Local Macbook
    or
    $ pip install -r requirements.txt # for GPU Server
    $ gunicorn main:app --bind=0.0.0.0:5000 --log-file=-

### How to run on Docker ###

    $ docker run -it -p 5000:5000 --rm <docker_hub_id>/mnist-api:cuda9.0-cudnn7

- -it: interative mode
- -p: expose port
- --rm: remove container after exiting