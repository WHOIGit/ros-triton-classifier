# Defines a 2D classification result.
#
# This result does not contain any position information. It is designed for
#   classifiers, which simply provide class probabilities given a source image.

Header header

# A list of class probabilities. This list need not provide a probability for
#   every possible class, just ones that are nonzero, or above some
#   user-defined threshold.
ObjectHypothesisWithClassName[] results
