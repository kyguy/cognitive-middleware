# Cognitive Middleware

Application configuration management for OpenShift via recurrent neural networks.(WIP)

Based on the work of [Andrej Karpathy](https://gist.github.com/karpathy/d4dee566867f8291f086), this code trains a recurrent neural network to memorize the character patterns found in [OpenShift application templates](https://github.com/jboss-openshift/application-templates), and leverages these learned patterns to detect and correct errors in template files.

## Error Detection and Correction
To scan a template file run:

    python3 scan.py <template_file> 

e.g.

    python3 scan.py demo/eap64-basic-s2i.json

> Note: This tool has only been tested and trained on [JBoss Middleware application templates](https://github.com/jboss-openshift/application-templates) and is still in early development.


## Train
To train the network for other configuration files, first concatenate a group of sample configuration files into one txt file and run:

    python3 rnn.py <txt_file>
e.g.

    python3 rnn.py data/input.txt

> Note: this code was taken from a small example gist of [Andrej Karpathy](https://gist.github.com/karpathy/d4dee566867f8291f086). Not siginificant alterations has been made towards optimization yet. Training takes a long time

