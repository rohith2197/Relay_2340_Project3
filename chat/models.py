from django.db import models

# Create your models here.
class QueryLog(models.Model):
    """One row per question. No question text is stored."""

    class Outcome(models.TextChoices):
        ANSWERED = "answered", "Answered"
        NO_MATCH = "no_match", "Not in sources"
        UNGROUNDED = "ungrounded", "Ungrounded (replaced)"
        DECLINED = "declined", "Declined"
        ERROR = "error", "Error"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    outcome = models.CharField(max_length=20, choices=Outcome.choices, db_index=True)
    response_time_ms = models.PositiveIntegerField()
    rewrite_ms = models.PositiveIntegerField(null=True, blank=True)
    retrieval_ms = models.PositiveIntegerField(null=True, blank=True)
    generation_ms = models.PositiveIntegerField(null=True, blank=True)
    status_code = models.PositiveSmallIntegerField(default=200)
    error_type = models.CharField(max_length=100, blank=True)
    model = models.CharField(max_length=100, blank=True)
    retrieved_chunks = models.PositiveSmallIntegerField(default=0)
    cited_sources = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    @property
    def is_error(self):
        return self.outcome == self.Outcome.ERROR

    def __str__(self):
        return f"{self.created_at:%Y-%m-%d %H:%M:%S} {self.outcome} {self.response_time_ms}ms"
