# Generated by Django 5.0.3 on 2024-03-31 14:35

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0002_hobby_delete_user"),
    ]

    operations = [
        migrations.DeleteModel(
            name="Hobby",
        ),
    ]
