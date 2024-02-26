from django.test import TestCase
from django.urls import reverse
from email_validator.models import ProcessedEmail
from email_validator.views import classify_email_machine_learning_based

class EmailClassificationTestCase(TestCase):
    def test_classify_email_logic_based(self):
        response = self.client.get(reverse('classify_email_logic_based'))
        self.assertEqual(response.status_code, 200)
        processed_emails_count = ProcessedEmail.objects.count()
        self.assertGreater(processed_emails_count, 0)

    def test_classify_email_machine_learning_based(self):
        test_details = [{'text': 'Test email 1', 'classification': 'not spam'},
                        {'text': 'Test email 2', 'classification': 'spam'},
                        {'text': 'Test email 3', 'classification': 'not spam'},
                        {'text': 'Test email 4', 'classification': 'spam'},
                        {'text': 'Test email 5', 'classification': 'not spam'}]
        
        response = classify_email_machine_learning_based(test_details)
        self.assertEqual(response.status_code, 200)
        
        processed_emails_count = ProcessedEmail.objects.count()
        self.assertGreater(processed_emails_count, 0)
