# Generated by Django 5.0.3 on 2024-04-18 18:59

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0003_choice"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="choice",
            name="age_range",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="distance",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="event_frequency",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="gender",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="group_size",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="notification_method",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="politics",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="race_ethnicity",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="religion",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="sexual_orientation",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="similarity_attribute",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="similarity_metric",
        ),
        migrations.RemoveField(
            model_name="choice",
            name="time_of_day",
        ),
        migrations.AddField(
            model_name="choice",
            name="categories",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
