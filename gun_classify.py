import numpy as np
import sounddevice as sd
import tensorflow as tf
from twilio.rest import Client

# Load the TensorFlow Lite model.
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up Twilio client.
account_sid = 'YOUR_ACCOUNT_SID' #Add This
auth_token = 'YOUR_AUTH_TOKEN' #ADD THIS
client = Client(account_sid, auth_token)
my_phone_number = '+1234567890'

# Define a function to listen for audio input and classify it.
def classify_audio(frames):
    # Convert audio to the input format expected by the model.
    input_data = np.array(frames).reshape(input_details[0]['shape'])
    # Run the model on the input data.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Get the index of the class with the highest probability.
    class_index = np.argmax(output_data)
    # Check if the class is a gunshot with 100% probability.
    if class_index == 0 and output_data[class_index] == 1.0:
        # Send a text message via Twilio.
        message = client.messages.create(
            to=my_phone_number, #ADD THIS
            from_='TWILIO_PHONE_NUMBER', #ADD THIS
            body='Gunshots detected at Substation 27!'
        )

# Start listening for audio input.
with sd.InputStream(callback=classify_audio):
    sd.sleep(1000000)
