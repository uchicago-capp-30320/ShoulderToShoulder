# Generated by Django 5.0.4 on 2024-05-03 17:25

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0002_eventsuggestion_user_id"),
    ]

    operations = [
        migrations.RenameField(
            model_name="eventsuggestion",
            old_name="hobby_category_art",
            new_name="hobby_category_arts_and_culture",
        ),
        migrations.RenameField(
            model_name="eventsuggestion",
            old_name="hobby_category_arts_and_crafts",
            new_name="hobby_category_crafting",
        ),
        migrations.RenameField(
            model_name="eventsuggestion",
            old_name="hobby_category_books",
            new_name="hobby_category_literature",
        ),
        migrations.RemoveField(
            model_name="eventsuggestion",
            name="hobby_category_movies",
        ),
        migrations.RemoveField(
            model_name="eventsuggestion",
            name="hobby_category_music",
        ),
        migrations.RemoveField(
            model_name="eventsuggestion",
            name="hobby_category_writing",
        ),
        migrations.AddField(
            model_name="eventsuggestion",
            name="hobby_category_community",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="eventsuggestion",
            name="hobby_category_food",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="eventsuggestion",
            name="hobby_category_history",
            field=models.BooleanField(default=False),
        ),
    ]