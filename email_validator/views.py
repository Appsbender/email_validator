import os

from django.http import HttpResponse, JsonResponse
from django.template import loader
from email_validator import settings
from email_validator.models import ProcessedEmail
from email_validator.utils import RANDOM_KEYWORDS

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from email_validator.services.csv_file_reader import read_csv
  
def main(request):
  template = loader.get_template('main.html')
  return HttpResponse(template.render())
  
def classify_content(text):
    for txt in RANDOM_KEYWORDS:
        if txt.lower() in text.lower():
            return 'spam'
    return 'not spam'  
    
#
# Let used static  files to serve CSV file for 
# uploading emails for the sake  of simplicity
#   
def classify_email_logic_based(request):
    dataset_path = os.path.join(settings.STATIC_ROOT, 'email_validator/csv/email_dataset.csv')
    details = read_csv(dataset_path)
    processed_data = [{'text': row.get('text', ''), 'classification': classify_content(row.get('text', ''))} for row in details]
    return classify_email_machine_learning_based(processed_data)
    
def classify_email_machine_learning_based(details):
    texts = [data['text'] for data in details]
    classifications = [data['classification'] for data in details]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post')

    # Convert data binary labels (0 for not spam, 1 for spam)
    binary_labels = [1 if label.lower() == 'spam' else 0 for label in classifications]

    # Split the datails to train
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, binary_labels, test_size=0.2, random_state=42)

    process = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    process.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Convert data to TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    # Train the model
    process.fit(train_dataset, epochs=10, validation_data=test_dataset)

    # Evaluate the process
    loss, accuracy = process.evaluate(test_dataset)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # save it
    for x in details:
        ProcessedEmail.objects.create(
            text=x['text'],
            classification=x['classification'],
            accuracy=accuracy,
            loss=loss
        )

    return JsonResponse({'status': 'Model trained successfully', 'accuracy': accuracy})
  