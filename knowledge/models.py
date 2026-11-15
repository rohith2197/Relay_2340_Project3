from django.db import models


class Document(models.Model):
    class DisasterType(models.TextChoices):
        FLOOD = "flood", "Flood"
        HURRICANE = "hurricane", "Hurricane"
        WILDFIRE = "wildfire", "Wildfire"
        TORNADO = "tornado", "Tornado"
        EARTHQUAKE = "earthquake", "Earthquake"
        GENERAL = "general", "General / multiple"

    class Phase(models.TextChoices):
        PREPAREDNESS = "preparedness", "Preparedness"
        RESPONSE = "response", "Response"
        RECOVERY = "recovery", "Recovery"
        GENERAL = "general", "General / multiple"

    class SourceType(models.TextChoices):
        MANUAL = "manual", "Added in admin"
        MANIFEST = "manifest", "Dataset manifest"
        UPLOAD = "upload", "File upload"
        SCRAPE = "scrape", "Web scrape"
        API = "api", "API"

    slug = models.SlugField(max_length=100, unique=True)
    title = models.CharField(max_length=255)
    source_url = models.URLField(max_length=500, blank=True)
    source_type = models.CharField(max_length=20, choices=SourceType.choices, default=SourceType.MANUAL)
    disaster_type = models.CharField(max_length=20, choices=DisasterType.choices, default=DisasterType.GENERAL)
    phase = models.CharField(max_length=20, choices=Phase.choices, default=Phase.GENERAL)
    is_active = models.BooleanField(default=True)

    raw_text = models.TextField()
    clean_text = models.TextField(blank=True, editable=False)
    content_hash = models.CharField(max_length=64, blank=True, editable=False, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["title"]

    def __str__(self):
        return self.title


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    position = models.PositiveIntegerField()
    text = models.TextField()

    class Meta:
        ordering = ["document_id", "position"]
        constraints = [
            models.UniqueConstraint(fields=["document", "position"], name="unique_chunk_position"),
        ]

    def __str__(self):
        return f"{self.document.slug} #{self.position}"