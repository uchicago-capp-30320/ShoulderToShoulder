# Generated by Django 5.0.4 on 2024-05-08 00:36

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0014_availability_day_of_week_availability_hour"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="availability",
            unique_together=set(),
        ),
        migrations.RemoveField(
            model_name="availability",
            name="calendar_id",
        ),
    ]