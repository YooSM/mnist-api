#!/usr/bin/env python
# -*-coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from mnist import model


x = tf.placeholder("float", [None, 784])
sess = tf.Session()

y, variables = model.convolution(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/model/model.ckpt")


def convolutional(input):
    return sess.run(y, feed_dict={x: input}).flatten().tolist()


app = Flask(__name__)
CORS(app)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    try:
        start_time = time.time()
        input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
        output = convolutional(input)
        elapsed_time = time.time() - start_time
        """
        TODO: kemi에 elapsed_time, message log 보내기
        logger.info({
            'message': "SUCCESS_MNIST",
            'elapsed_time': elapsed_time,
            'result': output.index(max(output))
        })
        """
        return jsonify(results=[output])
    except Exception as e:
        """
        TODO: watchtower에 에러 알람 보내기
        TODO: kemi에 에러 log 보내기
        logger.error({
            'message': "FAILED_MNIST",
            'exception': str(e)
        }, exc_info=True)
        """
        pass


@app.route('/')
def main():
    return "health check"


if __name__ == '__main__':
    app.run()
