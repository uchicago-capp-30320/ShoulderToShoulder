# Generated by Django 5.0.4 on 2024-04-26 16:39

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0014_alter_onboarding_distance_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="onboarding",
            name="address_line1",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name="onboarding",
            name="city",
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name="onboarding",
            name="event_frequency",
            field=models.CharField(
                blank=True,
                choices=[
                    ("Twice a week", "Twice a week"),
                    ("Once a week", "Once a week"),
                    ("Once every two weeks", "Once every two weeks"),
                    ("Once a month", "Once a month"),
                    ("Once every three months", "Once every three months"),
                ],
                max_length=100,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="onboarding",
            name="event_notification",
            field=models.CharField(
                choices=[
                    ("Email Only", "Email Only"),
                    ("Text Only", "Text Only"),
                    ("Email and Text", "Email and Text"),
                    ("None", "None"),
                ],
                default=True,
            ),
        ),
        migrations.AddField(
            model_name="onboarding",
            name="state",
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name="onboarding",
            name="zip_code",
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
    ]
