[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/gIcxkxvn)
# Objectives

The learning objectives of this assignment are to:
1. implement feed-forward prediction for a simple neural network
2. implement training via backpropagation for a simple neural network

NOTE: You are not allowed to use existing neural network frameworks 
that already implement functionality like forward and backward
propagation or training (such as PyTorch or TensorFlow). The purpose 
of this assignment is to provide your "initiation" to the details of 
the implementing a simple but complete neural network, with 
backpropagation gradient calculation and gradient descent. You should 
restrict yourself to only using data structures and functions in numpy, 
and with one exception, optionally a scipy function (expit). If you
want to use other packages, you must contact us ahead of time about
their use.

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [scipy](https://scipy.org/)
* [pytest](https://docs.pytest.org/)
* [pytest-timeout](https://pypi.org/project/pytest-timeout/)
* [pylint](https://pypi.org/project/pylint/)

# Check out the starter code

After accepting the assignment on GitHub Classroom, clone the newly created
repository to your local machine:
```
https://github.com/ml4ai-2025-spring-nn/pa2-<your-username>.git
```
You are now ready to begin working on the assignment.
You should do all your work in the default branch, `main`.

# Write your code

You will be editing the file `nn.py`.
**Do not add or edit any other files.**

You will implement a simple neural network with sigmoid activations at each layer.
This will include making predictions with a network via forward-propagation, and
training the network via gradient descent, with gradients calculated using
back-propagation.

You should read the documentation strings (docstrings) in each of the methods 
in `nn.py`, and implement the methods as described. You can import additional 
standard Python packages, and you also add other helper methods to the class
or functions to `nn.py` if desired. As noted above, if you with to include
non-standard Python packages, you much contact the instructors before you do so.

Write your code below the docstring of each method;
**do not delete or edit the docstrings**.

The following objects and functions may come in handy:
* [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
* [numpy.multiply](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
* [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)
* [numpy.ndarray.T](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html)
* [numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
* [scipy.special.expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)

# Test your code for correctness

The tests in `test_nn.py` check that each method behaves as expected.
To run all the provided tests, run ``pytest`` from the directory containing
``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 5 items

test_nn.py FFFFF                                                         [100%]

=================================== FAILURES ===================================
...
============================== 5 failed in 0.65s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 5 items

test_nn.py .....                                                         [100%]

============================== 5 passed in 0.47s ===============================
```

# Test your code for quality

In addition to the correctness tests, you should run `pylint nn.py` to check
for common code quality problems.
Pylint will check for adherence to
[standard Python style](https://www.python.org/dev/peps/pep-0008/),
good variable names, proper use of builtins like `enumerate`, etc.
If you use `scipy.special.expit`, you may also add the
`--extension-pkg-allow-list=scipy.special` option to silence a lint warning
from that package.
If there are no problems, you should see something like:
```
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```

# Submit your code

**Do not merge your pull request**

As you are working on the code, you should regularly `git commit` to save your
current changes locally.
You should also regularly `git push` to push all saved changes to the remote
repository on GitHub.
Make a habit of checking the "Feedback" pull request on the GitHub page for your
repository.
You should see all your pushed commits there, as well as the status of the
"checks".
If any correctness (pytest) or quality (pylint) tests are failing, you will see
"checks have failed" at the bottom of the pull request.
If you want to see exactly which tests have failed, click on the "Details" link.
When you have corrected all problems, you should see "All checks have passed"
at the bottom of the pull request.

You do not need to do anything beyond pushing your commits to Github to submit
your assignment.
The instructional team will grade the code of the "Feedback" pull request, and
make detailed comments there.

**Do not merge your pull request.**

# Grading

The points are allocated as follows:
* 80 points for passing all automated correctness (pytest) tests
* 10 points for passing all automated quality (pylint) tests
* 10 points for other quality issues:
using appropriate data structures,
using existing standard python library functions whenever appropriate,
minimizing code duplication,
giving variables meaningful names,
documenting complex pieces of code, etc.

Additionally, you will lose 5 points for each failure to follow instructions above, including:
* Editing files other than `nn.py`
* Adding any files or directories other than those mentioned above
* Deleting or editing the docstrings
* Merging your pull request
