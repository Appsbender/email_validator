from django.db import models
from django.utils import timezone

class ProcessedEmail(models.Model):
    id = models.AutoField(primary_key=True)
    text = models.TextField()
    classification = models.CharField(max_length=100)
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.text[:50]} - {self.classification}"
