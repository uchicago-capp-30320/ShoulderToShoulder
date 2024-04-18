# Generated by Django 5.0.3 on 2024-04-18 02:56

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="hobby",
            name="scenario_format",
            field=models.CharField(blank=True, default=None, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name="hobby",
            name="type",
            field=models.CharField(
                choices=[
                    ("Arts & Crafts", "Crafts"),
                    ("Books", "Books"),
                    ("Cooking/Baking", "Cooking"),
                    ("Exercise", "Exercise"),
                    ("Gaming", "Gaming"),
                    ("Music", "Music"),
                    ("Movies", "Movies"),
                    ("Outdoor Activities", "Outdoors"),
                    ("Art", "Art"),
                    ("Travel", "Travel"),
                    ("Writing", "Writing"),
                    ("Other", "Other"),
                ],
                default="Other",
                max_length=20,
            ),
        ),
    ]