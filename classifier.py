class Classifier:

    def __init__(self):
        """Use this to set and store the parameters of your 
        model that you will update and use to predict.
        """
        self.params = None

    def update(self, data):
        """Update the model based on new data.
        
        Args:
            data: a string containing the data in CSV
              format. Fields appear in the exact same
              order as they appear in the training data. 
              This CSV does not contain a header row.
        """
        raise NotImplementedError

    def predict(self, data):
        """Predict the controversiality of a new observation.

        Args:
            data: a string containing the data in CSV
              format. Fields are in the exact same
              order as they were in the training data.
              Some fields (e.g., ups, controversiality)
              will be blank. This CSV does not contain
              a header row.

        Returns:
            a list containing the predictions (0 or 1).
            There should be one prediction for each row
            of the input CSV.
        """
