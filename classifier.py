class Classifier:

    def __init__(self):
        """Use this to set and store the initial parameters that
        you will later update and use to predict.
        """
        self.params = None

    def update(self, data):
        """Update the model based on new data.
        
        Args:
            data: an array of Python dicts containing JSON
              data of one or more comments. The
              "controversiality" field is assumed to be 0 or 1.
        """
        raise NotImplementedError

    def predict(self, data):
        """Predict the controversiality of a new observation.

        Args:
            data: an array of Python dicts containing JSON
              data of one or more comments. The "upvotes",
              "downvotes", and "controversiality" fields
              should be assumed to be not set.
        """
