import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define a class for the Autoencoder
class Autoencoder:
    def __init__(self, input_size, hidden_units, learning_rate=0.01, epochs=50):
        # Initialize the Autoencoder with specified parameters
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases for encoding and decoding
        self.weights_encoder = np.random.randn(self.input_size, self.hidden_units)
        self.weights_decoder = np.random.randn(self.hidden_units, self.input_size)
        self.bias_encoder = np.zeros((1, self.hidden_units))
        self.bias_decoder = np.zeros((1, self.input_size))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, X):
        # Forward pass through the encoding layer
        return self.sigmoid(np.dot(X, self.weights_encoder) + self.bias_encoder)

    def backward_pass(self, hidden_activations, X):
        # Backward pass through the decoding layer, compute reconstruction loss
        reconstructed_data = self.sigmoid(np.dot(hidden_activations, self.weights_decoder) + self.bias_decoder)
        reconstruction_loss = np.mean((X - reconstructed_data) ** 2)

        # Compute gradients for updating weights and biases
        gradient_decoder = -(X - reconstructed_data) * reconstructed_data * (1 - reconstructed_data)
        gradient_encoder = np.dot(gradient_decoder, self.weights_decoder.T) * hidden_activations * (1 - hidden_activations)

        # Update weights and biases using gradient descent
        self.weights_decoder -= self.learning_rate * np.dot(hidden_activations.T, gradient_decoder)
        self.weights_encoder -= self.learning_rate * np.dot(X.T, gradient_encoder)
        self.bias_decoder -= self.learning_rate * np.sum(gradient_decoder, axis=0, keepdims=True)
        self.bias_encoder -= self.learning_rate * np.sum(gradient_encoder, axis=0, keepdims=True)

        return reconstruction_loss

    def train(self, X):
        # Training loop for the specified number of epochs
        for epoch in range(self.epochs):
            # Forward and backward passes, print reconstruction loss every 10 epochs
            hidden_activations = self.forward_pass(X)
            reconstruction_loss = self.backward_pass(hidden_activations, X)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Reconstruction Loss: {reconstruction_loss}")

    def predict(self, X):
        # Prediction using the trained Autoencoder on new data
        hidden_activations = self.forward_pass(X)
        return self.sigmoid(np.dot(hidden_activations, self.weights_decoder) + self.bias_decoder)

# Main function
def main():
    # Load data from a CSV file
    file_path = "/Users/igormol/Desktop/training_data.csv"
    data = pd.read_csv(file_path)

    # Separate features and target (assuming the last column is the target)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Autoencoder
    input_size = X_train_scaled.shape[1]
    hidden_units = 16
    autoencoder = Autoencoder(input_size, hidden_units)
    autoencoder.train(X_train_scaled)

    # Prediction on the test set
    reconstructed_data_test = autoencoder.predict(X_test_scaled)

    # Round values in the DataFrame to 2 decimal places for better readability
    X_test_rounded = np.round(X_test_scaled, decimals=2)
    reconstructed_data_rounded = np.round(reconstructed_data_test, decimals=2)

    # Create a DataFrame with rounded values for actual and predicted data
    result_table = pd.DataFrame(data=np.hstack((X_test_rounded, reconstructed_data_rounded)),
                                 columns=[f'Actual_{i}' for i in range(X_test_rounded.shape[1])] +
                                         [f'Predicted_{i}' for i in range(reconstructed_data_rounded.shape[1])])

    # Set the display format for float values and print the result table
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(result_table)

if __name__ == "__main__":
    main()

