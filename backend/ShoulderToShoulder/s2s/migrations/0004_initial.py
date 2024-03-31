# Generated by Django 5.0.3 on 2024-03-31 15:30

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("s2s", "0003_delete_hobby"),
    ]

    operations = [
        migrations.CreateModel(
            name="Hobby",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("description", models.TextField()),
                ("max_participants", models.IntegerField()),
                (
                    "type",
                    models.CharField(
                        choices=[
                            ("Sport", "Sport"),
                            ("Music", "Music"),
                            ("Art", "Art"),
                            ("Literature", "Literature"),
                            ("Science", "Science"),
                            ("Technology", "Technology"),
                            ("Crafts", "Crafts"),
                            ("Gaming", "Gaming"),
                            ("Cooking", "Cooking"),
                            ("Other", "Other"),
                        ],
                        default="Other",
                        max_length=20,
                    ),
                ),
            ],
        ),
    ]
