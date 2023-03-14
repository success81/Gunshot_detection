  GNU nano 5.4                                                                                                                                            gun_classify.py                                                                                                                                                     
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
account_sid = 'xxxxx'   
auth_token = 'xxxxx'     
client = Client(account_sid, auth_token)
my_phone_number = 'xxxx'

# Define a flag to track if a gunshot has already been detected.
gunshot_detected = False

# Define a function to listen for audio input and classify it.
def classify_audio(indata, frames, time, status):
    global gunshot_detected
    # Convert audio to the input format expected by the model.
    input_shape = (1, 44032)
    input_data = np.array(indata[:1376]).reshape(input_shape)
    # Run the model on the input data.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    # Get the index of the class with the highest probability.
    class_index = np.argmax(output_data)
    # Check if the class is a gunshot with low probability.
    if class_index == 2 and output_data[0, class_index] >= 0.99 and not gunshot_detected:
        # Send a text message via Twilio.
        message = client.messages.create(
            to=my_phone_number,
            from_='xxxx',
            body='Gunshots detected at Power Grid Substation 26!'
        )
        # Set the gunshot detected flag to True.
        gunshot_detected = True

# Start listening for audio input.
with sd.InputStream(callback=classify_audio, blocksize=(1376)):
    sd.sleep(1000000)

