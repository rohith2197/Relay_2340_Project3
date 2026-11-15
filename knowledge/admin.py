from django import forms
from django.contrib import admin, messages
from django.db.models import Count

from .ingestion import refresh_document
from .models import Chunk, Document
from .preprocessing import PreprocessingError, clean_text


class DocumentAdminForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = "__all__"

    def clean_raw_text(self):
        raw = self.cleaned_data["raw_text"]
        try:
            clean_text(raw)
        except PreprocessingError as exc:
            raise forms.ValidationError(str(exc))
        return raw


class ChunkInline(admin.TabularInline):
    model = Chunk
    extra = 0
    can_delete = False
    fields = ("position", "text")
    readonly_fields = ("position", "text")

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    form = DocumentAdminForm
    list_display = ("title", "disaster_type", "phase", "source_type", "is_active", "chunk_count", "updated_at")
    list_filter = ("disaster_type", "phase", "source_type", "is_active")
    search_fields = ("title", "slug", "source_url")
    prepopulated_fields = {"slug": ("title",)}
    readonly_fields = ("clean_text", "content_hash", "created_at", "updated_at")
    inlines = [ChunkInline]
    actions = ["reprocess_documents", "activate_documents", "deactivate_documents"]

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_chunk_count=Count("chunks"))

    @admin.display(description="Chunks", ordering="_chunk_count")
    def chunk_count(self, obj):
        return obj._chunk_count

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        refresh_document(obj)

    @admin.action(description="Re-process selected documents (re-clean and re-chunk)")
    def reprocess_documents(self, request, queryset):
        done = 0
        for document in queryset:
            try:
                refresh_document(document)
                done += 1
            except PreprocessingError as exc:
                self.message_user(request, f"{document.title}: {exc}", level=messages.ERROR)
        self.message_user(request, f"Re-processed {done} document(s).")

    @admin.action(description="Include selected documents in answers")
    def activate_documents(self, request, queryset):
        self.message_user(request, f"Activated {queryset.update(is_active=True)} document(s).")

    @admin.action(description="Exclude selected documents from answers")
    def deactivate_documents(self, request, queryset):
        self.message_user(request, f"Deactivated {queryset.update(is_active=False)} document(s).")